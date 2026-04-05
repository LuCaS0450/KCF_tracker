import numpy as np
import cv2
import fhog


class KCFTracker:
    def __init__(self):
        self.interp_factor = 0.03
        self.sigma = 0.5
        self.lambda_ = 1e-4
        self.cell_size = 4
        self.padding = 2.5
        self.psr_threshold = 5.5

        # Scale adaptation parameters (升级为标准的 7 尺度搜索)
        # ===================================================
        # 1. 步长调小：因为搜索变密集了，从 1.05 缩小到 1.02，让框的变化更平滑
        self.scale_step = 1.02

        # 2. 扩充为 7 个尺度因子：[0.942, 0.961, 0.980, 1.000, 1.020, 1.040, 1.061]
        self.scale_factors = [self.scale_step ** i for i in range(-3, 4)]

        self.scale_interp_factor = 0.02
        self.current_scale_factor = 1.0

        # 3. 匹配 7 个惩罚权重：距离 1.0 越远的尺度，惩罚力度越大
        # 这能有效防止目标稍微侧身，追踪框就瞬间剧烈变大或缩小（抑制尺度抖动）
        self.scale_penalties = [0.970, 0.985, 0.995, 1.0, 0.995, 0.985, 0.970]

    def init(self, image, roi):
        self._roi = list(roi)
        x, y, w, h = roi
        self.target_sz = (int(max(2, w)), int(max(2, h)))
        self.pos = [y + h / 2.0, x + w / 2.0]  # [cy, cx]

        self._init_sizes()

        # Save base sizes for reference during scale updates
        self.base_target_sz = self.target_sz
        self.base_window_sz = self.window_sz

        # Cosine window in cell space
        self._window = np.outer(np.hanning(self.feat_sz[0]), np.hanning(self.feat_sz[1])).astype(np.float32)

        # Gaussian label in cell space
        output_sigma = np.sqrt(self.target_sz[0] * self.target_sz[1]) / self.cell_size * 0.125
        rs, cs = np.ogrid[:self.feat_sz[0], :self.feat_sz[1]]
        rs = rs - self.feat_sz[0] // 2
        cs = cs - self.feat_sz[1] // 2
        y_cell = np.exp(-0.5 * (rs * rs + cs * cs) / (output_sigma * output_sigma + 1e-6)).astype(np.float32)
        self.yf = np.fft.fft2(np.fft.ifftshift(y_cell))

        # Initial model
        x_patch = self.get_subwindow(image, self.pos, self.window_sz)
        x = self.get_features(x_patch)  # Hc x Wc x C
        xf = np.fft.fft2(x, axes=(0, 1))
        self.model_xf = xf
        self.model_alphaf = self.training(xf, self.yf)

    def _init_sizes(self):
        # Search window in pixel coordinates
        win_w = int(round(self.target_sz[0] * self.padding))
        win_h = int(round(self.target_sz[1] * self.padding))
        win_w = max(self.cell_size * 2, (win_w // self.cell_size) * self.cell_size)
        win_h = max(self.cell_size * 2, (win_h // self.cell_size) * self.cell_size)
        self.window_sz = (win_w, win_h)

        # Feature map size (cell coordinates)
        self.feat_sz = (self.window_sz[1] // self.cell_size, self.window_sz[0] // self.cell_size)

    def update(self, image):
        # Scale adaptation: test at multiple scales
        best_scale_factor = self.current_scale_factor
        best_score = -1.0
        best_r, best_c = 0, 0
        best_responses = None
        best_psr = -1.0

        for i, scale_multiplier in enumerate(self.scale_factors):
            test_scale_factor = self.current_scale_factor * scale_multiplier

            # Calculate extraction window for this scale
            test_window_sz = (
                int(self.base_window_sz[0] * test_scale_factor),
                int(self.base_window_sz[1] * test_scale_factor)
            )

            # Prevent window from becoming too small or impossibly large
            if test_window_sz[0] < self.cell_size * 4 or test_window_sz[1] < self.cell_size * 4:
                continue
            if test_window_sz[0] > image.shape[1] * 1.5 or test_window_sz[1] > image.shape[0] * 1.5:
                continue

            # Extract patch and resize to BASE window size
            z_patch = self.get_subwindow(image, self.pos, test_window_sz, out_sz=self.base_window_sz)
            z = self.get_features(z_patch)
            zf = np.fft.fft2(z, axes=(0, 1))

            kzf = self.gaussian_correlation(zf, self.model_xf)
            responses = np.real(np.fft.ifft2(self.model_alphaf * kzf))

            r, c = np.unravel_index(np.argmax(responses), responses.shape)

            max_response = responses[r, c]

            # Calculate PSR properly
            mean_response = np.mean(responses)
            std_response = np.std(responses)
            psr = (max_response - mean_response) / (std_response + 1e-6)

            # Use penalized max_response for scale selection.
            # In standard SAMF, max_response is preferred over PSR for scale selection
            # because PSR is too volatile across slight positional shifts.
            score = max_response * self.scale_penalties[i]

            if score > best_score:
                best_score = score
                best_psr = psr
                best_scale_factor = test_scale_factor
                best_r, best_c = r, c
                best_responses = responses

        r, c = best_r, best_c
        psr = best_psr

        if best_responses is not None:
            if r > best_responses.shape[0] // 2:
                r -= best_responses.shape[0]
            if c > best_responses.shape[1] // 2:
                c -= best_responses.shape[1]
        else:
            r = c = 0

        # Cell shift -> pixel shift, using the optimal scale
        self.pos[0] += (r * self.cell_size) * best_scale_factor
        self.pos[1] += (c * self.cell_size) * best_scale_factor

        # Instantly update tracking scale and bounding box size
        if psr > self.psr_threshold:
            # Bug fix: Previously we interpolated using scale_interp_factor=0.02.
            # Since best_scale_factor only bounds by 1.05 max, interpolating it by 0.02
            # caused the box to only grow by 0.1% per frame, which appeared completely invisible.
            # Standard SAMF directly accepts the tested scale multiplier.
            scale_weight = 0.15  # allowing it to update fast enough
            self.current_scale_factor = (1.0 - scale_weight) * self.current_scale_factor + scale_weight * best_scale_factor

            self.target_sz = (
                self.base_target_sz[0] * self.current_scale_factor,
                self.base_target_sz[1] * self.current_scale_factor
            )

        # Re-train using a patch perfectly centered at NEW position and exact smoothed scale
        train_window_sz = (
            int(self.base_window_sz[0] * self.current_scale_factor),
            int(self.base_window_sz[1] * self.current_scale_factor)
        )
        x_patch = self.get_subwindow(image, self.pos, train_window_sz, out_sz=self.base_window_sz)

        x = self.get_features(x_patch)
        xf = np.fft.fft2(x, axes=(0, 1))
        alphaf = self.training(xf, self.yf)

        if psr > self.psr_threshold:
            self.model_alphaf = (1.0 - self.interp_factor) * self.model_alphaf + self.interp_factor * alphaf
            self.model_xf = (1.0 - self.interp_factor) * self.model_xf + self.interp_factor * xf

        self._roi = [
            float(self.pos[1] - self.target_sz[0] / 2.0),
            float(self.pos[0] - self.target_sz[1] / 2.0),
            float(self.target_sz[0]),
            float(self.target_sz[1]),
        ]
        return self._roi

    def gaussian_correlation(self, xf, yf):
        # xf, yf: H x W x C (frequency domain)
        H, W, C = xf.shape

        # Correct scale: get spatial domain sum of squares
        xx = np.sum(np.abs(xf) ** 2) / (H * W)
        yy = np.sum(np.abs(yf) ** 2) / (H * W)

        # Cross-correlation summed over channels -> H x W
        xyf = np.sum(xf * np.conj(yf), axis=2)
        xy = np.real(np.fft.ifft2(xyf))

        # We only average the Euclidean distance over the spatial dimensions.
        # Because features are L2 normalized per cell, xx and yy are exactly 1.0 per cell.
        # Bounding dist properly between 0.0 and 2.0 to give a sharp spatial Gaussian peak.
        dist = np.maximum(0.0, (xx + yy - 2.0 * xy) / (H * W))

        k = np.exp(-dist / (self.sigma ** 2))
        return np.fft.fft2(k)

    def training(self, xf, yf):
        # yf: H x W (frequency), kf: H x W
        kf = self.gaussian_correlation(xf, xf)
        return yf / (kf + self.lambda_)

    def get_features(self, image):
        """
        Extract 31-dim Felzenszwalb HOG features from local fhog.py.
        Output: Hc x Wc x 31 float32
        """
        # 1. 确保基础图像块大小准确
        if image.shape[1] != self.window_sz[0] or image.shape[0] != self.window_sz[1]:
            image = cv2.resize(image, self.window_sz, interpolation=cv2.INTER_LINEAR)

        # 2. 核心修复：解决 fhog 底层丢弃边界导致的中心偏移问题
        # 在图像四周各 Pad 一个 cell_size 大小的边缘（使用边缘复制）
        pad = self.cell_size
        image_padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        # 3. 转换数据类型，准备送入 fhog
        img = image_padded.astype(np.float32, copy=False)
        # （注：fhog.py 内部已经自带了 / 255.0 的归一化逻辑，这里无需重复处理）

        # 4. 提取特征
        # 得益于上面的 padding，此时 fhog 吐出的 feat 尺寸完美等于 (Hc, Wc, 31)
        feat = fhog.fhog(img, self.cell_size)

        # 5. 直接乘以余弦窗压制边缘高频噪声，完全不需要切片或补零
        feat *= self._window[:, :, None]

        return feat.astype(np.float32, copy=False)

    def get_subwindow(self, image, pos, window_sz, out_sz=None):
        if out_sz is None:
            out_sz = window_sz

        h, w = image.shape[:2]
        cy, cx = int(round(pos[0])), int(round(pos[1]))
        wy, wx = window_sz[1], window_sz[0]

        top = cy - wy // 2
        bottom = top + wy
        left = cx - wx // 2
        right = left + wx

        pt_top = max(0, top)
        pt_bottom = min(h, bottom)
        pt_left = max(0, left)
        pt_right = min(w, right)

        # Padding sizes
        pad_top = pt_top - top
        pad_bottom = bottom - pt_bottom
        pad_left = pt_left - left
        pad_right = right - pt_right

        if pt_top < pt_bottom and pt_left < pt_right:
            valid_patch = image[pt_top:pt_bottom, pt_left:pt_right]
            patch = cv2.copyMakeBorder(valid_patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        else:
            # Completely out of bounds (edge case)
            patch = np.zeros((wy, wx, 3), dtype=np.uint8)

        if out_sz != window_sz:
            patch = cv2.resize(patch, (out_sz[0], out_sz[1]))

        return patch

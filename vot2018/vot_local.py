"""
简易版 vot.py (TraX 通信协议桥梁) - 修正大小写版
"""
import sys
import collections
import trax
import trax.server

# 定义 KCF 认识的矩形框格式
Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])

class VOT(object):
    def _extract_region(self, request):
        """从 TraX 请求中提取初始化区域。"""
        if not request.objects or len(request.objects) == 0:
            return None

        obj = request.objects[0]

        # objects 中的元素可能是 tuple (Rectangle, properties) 或直接的 Rectangle 对象
        if isinstance(obj, tuple):
            rect_obj = obj[0]
        else:
            rect_obj = obj

        if hasattr(rect_obj, 'bounds'):
            x, y, w, h = rect_obj.bounds()
            return type('Region', (), {
                'x': x,
                'y': y,
                'width': w,
                'height': h
            })()

        raise Exception(f"Cannot understand object format: {obj}, type: {type(obj)}")

    def __init__(self, region_format):
        # 【核心修复】：使用大写开头的类名 trax.Region 和 trax.Image
        # 只支持 RECTANGLE 类型
        self._trax = trax.server.Server([trax.Region.RECTANGLE], [trax.Image.PATH])

        request = self._trax.wait()
        if request.type == 'quit':
            sys.exit(0)

        # 获取图像路径
        img_path = None
        if isinstance(request.image, dict):
            image_obj = list(request.image.values())[0]
            if hasattr(image_obj, 'path'):
                img_path = image_obj.path if not callable(image_obj.path) else image_obj.path()
            else:
                img_path = str(image_obj)
        elif hasattr(request.image, 'path'):
            img_path = request.image.path if not callable(request.image.path) else request.image.path()
        else:
            img_path = str(request.image)

        # 去除 'file://' 前缀如果存在的话
        if isinstance(img_path, str) and img_path.startswith('file://'):
            img_path = img_path[7:]
            if sys.platform == 'win32' and img_path.startswith('/') and len(img_path) > 2 and img_path[2] == ':':
                img_path = img_path[1:]

        self._image = img_path
        self._pending_region = None

        self._region = self._extract_region(request)
        if self._region is None:
            raise Exception("No initialization region provided")

    def region(self):
        # 获取第一帧的初始框
        return Rectangle(self._region.x, self._region.y, self._region.width, self._region.height)

    def new_region(self):
        """获取并清空一次性的重初始化区域。"""
        if self._pending_region is None:
            return None

        region = Rectangle(
            self._pending_region.x,
            self._pending_region.y,
            self._pending_region.width,
            self._pending_region.height,
        )
        self._pending_region = None
        return region

    def frame(self):
        # 获取当前帧的图片路径
        if hasattr(self, '_image'):
            img = self._image
            del self._image
            # 如果是 FileImage 对象，需要提取路径
            if hasattr(img, 'path'):
                # path 可能是一个方法或属性
                if callable(img.path):
                    return img.path()
                else:
                    return img.path
            elif isinstance(img, str):
                return img
            else:
                return str(img)

        request = self._trax.wait()
        if request.type == 'quit':
            return None

        if request.type == 'initialize':
            self._pending_region = self._extract_region(request)

        # 处理 image 字段
        img_path = None
        
        if isinstance(request.image, dict):
            # 字典格式: {'color': FileImage object} 或类似
            # 提取第一个值（FileImage 对象）
            image_obj = list(request.image.values())[0]
            
            # 现在从 FileImage 对象获取路径
            if hasattr(image_obj, 'path'):
                try:
                    img_path = image_obj.path if not callable(image_obj.path) else image_obj.path()
                except Exception as e:
                    img_path = str(image_obj)
            else:
                img_path = str(image_obj)
        elif hasattr(request.image, 'path'):
            try:
                img_path = request.image.path if not callable(request.image.path) else request.image.path()
            except Exception as e:
                img_path = str(request.image)
        elif isinstance(request.image, bytes):
            # 字节字符串
            img_path = request.image.decode('utf-8')
        elif isinstance(request.image, str):
            # 普通字符串
            img_path = request.image
        else:
            # 其他类型，尝试转换
            img_path = str(request.image)
        
        # 处理 "File resource at '...'" 格式
        if isinstance(img_path, str) and img_path.startswith("File resource at '"):
            # 提取引号中的实际路径
            import re
            match = re.search(r"'([^']+)'", img_path)
            if match:
                img_path = match.group(1)
                # 如果是字节字符串表示，解码
                if img_path.startswith("b'"):
                    img_path = img_path[2:-1]  # 移除 b' 和末尾的 '

        # 去除 'file://' 前缀如果存在的话
        if isinstance(img_path, str) and img_path.startswith('file://'):
            img_path = img_path[7:]
            if sys.platform == 'win32' and img_path.startswith('/') and len(img_path) > 2 and img_path[2] == ':':
                img_path = img_path[1:]
        
        # 确保返回的是字符串
        if not isinstance(img_path, str):
            img_path = str(img_path)

        return img_path

    def report(self, region):
        # 汇报预测的矩形框给裁判
        if isinstance(region, Rectangle):
            # 【核心修复】：使用 trax.Rectangle.create 并包装为 (region, properties) 元组
            rect = trax.Rectangle.create(region.x, region.y, region.width, region.height)
            self._trax.status([(rect, {})])
        elif isinstance(region, list) and len(region) == 0:
            # 如果跟丢了，返回空列表（表示跟踪失效）
            self._trax.status([])
        else:
            # 其他情况也视为失效
            self._trax.status([])
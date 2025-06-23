from .user_service import *
from .text3d_service import Text3DService
from .img3d_service import Img3DService
from .textimg3d_service import TextImg3DService
from .unico3d_service import Unico3DService
from .multiimg3d_service import MultiImg3DService
from .boceto3d_service import Boceto3DService

text3d_service = Text3DService()
img3d_service = Img3DService()
textimg3d_service = TextImg3DService()
unico3d_service = Unico3DService()
multiimg3d_service = MultiImg3DService()
boceto3d_service = Boceto3DService()

SERVICE_INSTANCE_MAP = {
    'Texto3D': text3d_service,
    'Imagen3D': img3d_service,
    'TextImg3D': textimg3d_service,
    'Unico3D': unico3d_service,
    'MultiImagen3D': multiimg3d_service,
    'Boceto3D': boceto3d_service
}
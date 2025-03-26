import pkgutil
import importlib
import inspect

__all__ = []

# 현재 패키지(__init__.py가 위치한 곳)의 하위 모듈들을 순회합니다.
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    module = importlib.import_module("." + module_name, package=__name__)
    # 하위 모듈 내의 모든 멤버를 검사합니다.
    for name, obj in inspect.getmembers(module):
        # _ 로 시작하지 않는 공개 객체 중, 클래스나 함수만 선택합니다.
        if not name.startswith("_") and (
            inspect.isclass(obj) or inspect.isfunction(obj)
        ):
            globals()[name] = obj
            __all__.append(name)

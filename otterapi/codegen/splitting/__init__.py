"""Module splitting functionality for OtterAPI.

This package provides the infrastructure for splitting generated endpoints
into multiple organized Python modules based on configurable strategies.

Classes:
    ModuleMapResolver: Resolves endpoint paths to target modules.
    ModuleTree: Tree structure representing the module hierarchy.
    ModuleTreeBuilder: Builds module trees from endpoints and configuration.
    SplitModuleEmitter: Emits split modules to the filesystem.
    ResolvedModule: Result of resolving an endpoint to a module.
"""

from otterapi.codegen.splitting.emitter import SplitModuleEmitter
from otterapi.codegen.splitting.resolver import ModuleMapResolver, ResolvedModule
from otterapi.codegen.splitting.tree import ModuleTree, ModuleTreeBuilder

__all__ = [
    'ModuleMapResolver',
    'ResolvedModule',
    'ModuleTree',
    'ModuleTreeBuilder',
    'SplitModuleEmitter',
]

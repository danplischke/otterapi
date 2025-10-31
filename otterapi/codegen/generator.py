import ast
import http
import json
import logging
import os
import py_compile
import tempfile

import httpx
from openapi_pydantic import Operation
from openapi_pydantic.v3.parser import OpenAPIv3
from pydantic import TypeAdapter
from upath import UPath

from otterapi.codegen.ast_utils import _all, _assign, _call, _name, _union_expr
from otterapi.codegen.endpoints import async_request_fn, request_fn
from otterapi.codegen.openapi_processor import OpenAPIProcessor
from otterapi.codegen.type_generator import Endpoint, Parameter, Type, TypeGen
from otterapi.codegen.utils import (
    is_url,
    sanitize_identifier,
    sanitize_parameter_field_name,
)
from otterapi.config import DocumentConfig

HTTP_METHODS = [method.value.lower() for method in http.HTTPMethod]


class Codegen(OpenAPIProcessor):
    def __init__(self, config: DocumentConfig):
        super().__init__(None)
        self.config = config
        self.typegen: TypeGen | None = None

    def _load_schema(self):
        content = b''
        if is_url(self.config.source):
            response = httpx.get(self.config.source)
            response.raise_for_status()
            content = response.json()
        else:
            with open(self.config.source, 'rb') as f:
                content = json.loads(f.read())
        self.openapi = TypeAdapter(OpenAPIv3).validate_python(content)
        self.typegen = TypeGen(self.openapi)

    def _get_response_models_by_status_code(
        self, operation: Operation
    ) -> tuple[list[int] | None, Type | None]:
        if not operation.responses:
            return None, None

        types: dict[int, Type] = {}
        for status_code, response in operation.responses.items():
            if response.content:
                for content_type, media_type in response.content.items():
                    if media_type.media_type_schema:
                        if status_code in types:
                            raise ValueError(
                                'Multiple response schemas for the same status code are not supported'
                            )
                        types[int(status_code)] = self.typegen.schema_to_type(
                            media_type.media_type_schema,
                            base_name=f'{sanitize_identifier(operation.operationId)}Response',
                        )

        type_len = len(types)
        if type_len == 0:
            return None, None
        elif type_len == 1:
            return list(types.keys()), next(iter(types.values()))
        else:
            type_values = list(types.values())
            union_type = Type(
                None,
                None,
                annotation_ast=_union_expr([t.annotation_ast for t in type_values]),
                implementation_ast=None,
                type='primitive',
            )
            # Aggregate imports from all types
            union_type.copy_imports_from_sub_types(type_values)
            union_type.add_annotation_import('typing', 'Union')
            return list(types.keys()), union_type

    def _get_param_model(self, operation: Operation) -> list[Parameter]:
        params = []
        for param in operation.parameters or []:
            param_type = None
            if param.param_schema:
                param_type = self.typegen.schema_to_type(param.param_schema)

            params.append(
                Parameter(
                    name=param.name,
                    name_sanitized=sanitize_parameter_field_name(param.name),
                    location=param.param_in,  # query, path, header, cookie
                    required=param.required or False,
                    type=param_type,
                    description=param.description,
                )
            )

        if operation.requestBody:
            body, _ = self._resolve_reference(operation.requestBody)
            if body.content:
                for content_type, media_type in body.content.items():
                    if content_type != 'application/json':
                        logging.warning(
                            f'Skipping non-JSON request body content type: {content_type}'
                        )
                        continue

                    if media_type.media_type_schema:
                        body_type = self.typegen.schema_to_type(
                            media_type.media_type_schema,
                            base_name=f'{sanitize_identifier(operation.operationId)}RequestBody',
                        )
                        params.append(
                            Parameter(
                                name='body',
                                name_sanitized='body',
                                location='body',
                                required=body.required or False,
                                type=body_type,
                                description=body.description,
                            )
                        )

        return params

    def _generate_endpoint(self, path: str, method: str, operation: Operation):
        fn_name = (
            operation.operationId
            or f'{method}_{path.replace("/", "_").replace("{", "").replace("}", "")}'
        )
        async_fn_name = f'a{fn_name}'

        parameters = self._get_param_model(operation)
        supported_status_codes, response_model = (
            self._get_response_models_by_status_code(operation)
        )

        async_fn, async_imports = async_request_fn(
            name=async_fn_name,
            method=method,
            path=path,
            response_model=response_model,
            docs=operation.description,
            parameters=parameters,
            supported_status_codes=supported_status_codes,
        )

        sync_fn, imports = request_fn(
            name=fn_name,
            method=method,
            path=path,
            response_model=response_model,
            docs=operation.description,
            parameters=parameters,
            supported_status_codes=supported_status_codes,
        )

        ep = Endpoint(
            sync_ast=sync_fn,
            sync_fn_name=fn_name,
            async_fn_name=async_fn_name,
            async_ast=async_fn,
            name=fn_name,
        )

        ep.add_imports([imports, async_imports])

        for param in parameters:
            if param.type and param.type.annotation_imports:
                ep.add_imports([param.type.annotation_imports])

        return ep

    def _generate_endpoints(self):
        endpoints: list[Endpoint] = []
        for path, path_item in self.openapi.paths.items():
            for method in HTTP_METHODS:
                operation = getattr(path_item, method, None)
                if operation:
                    ep = self._generate_endpoint(path, method, operation)
                    endpoints.append(ep)
        return endpoints

    def _generate_file(self, body: list[ast.stmt], path: UPath) -> None:
        mod = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(mod)

        file_content = ast.unparse(mod)

        with tempfile.NamedTemporaryFile() as f:
            f.write(file_content.encode('utf-8'))
            f.flush()
            # check if the generated file is valid python
            py_compile.compile(f.name)

        with open(str(path), 'wb') as f:
            f.write(file_content.encode('utf-8'))

    def _generate_endpoint_file(
        self, path: UPath, models_file: UPath, endpoints: list[Endpoint]
    ):
        from otterapi.codegen.endpoints import base_async_request_fn, base_request_fn

        baseurl = None
        if not self.openapi.servers or self.config.base_url:
            baseurl = self.config.base_url
        elif self.openapi.servers:
            if len(self.openapi.servers) > 1:
                raise ValueError(
                    'Multiple servers are not supported. Set the base_url in the config.'
                )

            # TODO: handle variables
            baseurl = self.openapi.servers[0].url

        if not baseurl:
            raise ValueError(
                'No base url provided. Make sure you specify the base_url in the otterapi config or the OpenAPI document contains a valid servers section'
            )

        body: list[ast.stmt] = [
            _assign(_name('BASE_URL'), ast.Constant(baseurl)),
            _assign(_name('T'), _call(_name('TypeVar'), [ast.Constant('T')])),
        ]
        imports: dict[str, set[str]] = {}
        model_names: set[str] = set()

        # Add base request functions
        sync_base_fn, sync_base_imports = base_request_fn()
        async_base_fn, async_base_imports = base_async_request_fn()

        body.append(sync_base_fn)
        body.append(async_base_fn)

        # Collect imports from base functions
        for module, names in sync_base_imports.items():
            if module not in imports:
                imports[module] = set()
            imports[module].update(names)

        for module, names in async_base_imports.items():
            if module not in imports:
                imports[module] = set()
            imports[module].update(names)

        endpoint_names: set[str] = set()
        # Add endpoint functions and collect their imports
        for endpoint in endpoints:
            endpoint_names.update([endpoint.sync_fn_name, endpoint.async_fn_name])
            body.append(endpoint.sync_ast)
            body.append(endpoint.async_ast)
            for module, names in endpoint.imports.items():
                if module not in imports:
                    imports[module] = set()
                imports[module].update(names)

        # Collect all model names used in endpoints (from typegen.types)
        for type_name, type_ in self.typegen.types.items():
            if type_.name and type_.implementation_ast:
                model_names.add(type_.name)

        body.insert(0, _all(sorted(endpoint_names)))

        # Add import for models from the generated models file
        if model_names:
            model_import = ast.ImportFrom(
                module=self.config.models_import_path or models_file.stem,
                names=[
                    ast.alias(name=name, asname=None) for name in sorted(model_names)
                ],
                level=1 if not self.config.models_import_path else 0,  # relative import
            )
            body.insert(0, model_import)

        # Add all other imports at the beginning
        for module, names in sorted(imports.items(), reverse=True):
            import_stmt = ast.ImportFrom(
                module=module,
                names=[ast.alias(name=name, asname=None) for name in sorted(names)],
                level=0,
            )
            body.insert(0, import_stmt)

        self._generate_file(body, path)

    def _generate_models_file(self, path: UPath):
        assert self.typegen is not None

        body: list[ast.stmt] = []
        imports: dict[str, set[str]] = {}

        all_names = set()

        for type_name, type_ in self.typegen.types.items():
            if type_.implementation_ast:
                body.append(type_.implementation_ast)
                if type_.name:
                    all_names.add(type_.name)

                # Collect imports from implementation
                for module, names in type_.implementation_imports.items():
                    if module not in imports:
                        imports[module] = set()
                    imports[module].update(names)

                # Collect imports from annotations (List, Dict, Any, Union, Optional, etc.)
                for module, names in type_.annotation_imports.items():
                    if module not in imports:
                        imports[module] = set()
                    imports[module].update(names)

        body.insert(0, _all(sorted(all_names)))

        # Add all imports at the beginning
        for module, names in sorted(imports.items(), reverse=True):
            import_stmt = ast.ImportFrom(
                module=module,
                names=[ast.alias(name=name, asname=None) for name in sorted(names)],
                level=0,
            )
            body.insert(0, import_stmt)

        self._generate_file(body, path)

    def _generate_init_file(self, directory: UPath):
        init_file = directory / '__init__.py'
        if not init_file.exists():
            init_file.touch()

    def generate(self):
        self._load_schema()

        assert self.openapi is not None

        if not self.openapi.paths:
            raise ValueError('OpenAPI spec has no paths to generate endpoints from')

        directory = UPath(self.config.output)
        directory.mkdir(parents=True, exist_ok=True)

        if not os.access(str(directory), os.W_OK):
            raise RuntimeError(f'Directory {directory} is not writable')

        endpoints = self._generate_endpoints()

        models_file = directory / self.config.models_file
        self._generate_models_file(models_file)

        endpoints_file = directory / self.config.endpoints_file
        self._generate_endpoint_file(endpoints_file, models_file, endpoints)

        self._generate_init_file(directory)

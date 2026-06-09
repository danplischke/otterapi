# vulture whitelist — symbols that are "used" by framework magic, not by direct call.
# Keep each entry to the narrowest scope and document why it's here.
#
# Format: bare name for module-level symbols, _.name for class members.

_ = lambda *args, **kwargs: None  # noqa: E731

# ---------------------------------------------------------------------------
# CLI entry points  (registered by @app.command() / @<group>.command())
# ---------------------------------------------------------------------------
main  # otterapi/cli.py — click command
init  # otterapi/cli.py — click command
validate  # otterapi/cli.py — click command

# ---------------------------------------------------------------------------
# Pydantic field validators  (called by pydantic, not by application code)
# ---------------------------------------------------------------------------

# config.py
_.normalize_style
_.normalize_default_style
_.normalize_strategy
_.normalize_module_map_before
_.validate_source
_.validate_output
_.validate_filename
_.validate_documents

# config.py — pydantic Field(...) declarations assigned to class-level names;
# pydantic accesses them via model.__fields__, not direct name lookup.
format_output
validate_output
create_py_typed

# v2.py
_.validate_path_required
_.validate_and_convert_responses
_.validate_and_convert_paths

# ---------------------------------------------------------------------------
# Enum members  (accessed as EnumClass.MEMBER, not by bare name)
# ---------------------------------------------------------------------------

# v2.py — SchemeType
_.HTTPS
_.WS
_.WSS

# v2.py — PrimitiveType
_.STRING
_.NUMBER
_.INTEGER
_.BOOLEAN

# v3.py / v3_1.py / v3_2.py — Type enum
_.boolean
_.integer
_.number

# v3.py / v3_1.py — ParameterLocation enum
_.query
_.cookie

# v3.py / v3_1.py — ParameterStyle enum
_.matrix
_.spaceDelimited
_.pipeDelimited
_.deepObject

# v3_1.py extras
_.prefixItems
_.patternProperties
_.pathItems

# ---------------------------------------------------------------------------
# Pydantic model fields with Literal annotations
# (vulture sees the class-level assignment but not the runtime field access)
# ---------------------------------------------------------------------------

# v2.py OAuth2 security models and Swagger root
_.flow
_.swagger

# v3_2.py — License model and JSON Schema extension fields
_.identifier
_.contains
_.propertyNames
_.contentMediaType
_.contentEncoding
_.if_
_.then
_.else_
_.dependentSchemas

# ---------------------------------------------------------------------------
# Runtime template files (otterapi/codegen/runtime/)
# These files are copied verbatim into generated client packages.
# They are not imported by the library itself — their symbols are only used
# in user-generated code at runtime.
# ---------------------------------------------------------------------------
OffsetPaginationConfig
CursorPaginationConfig
PagePaginationConfig
export_async

# ---------------------------------------------------------------------------
# Golden fixture files (otterapi/tests/fixtures/golden/)
# Generated pydantic model fields — accessed as instance attributes, not
# by bare name.
# ---------------------------------------------------------------------------
_.kind
_.bark
_.meow

# ---------------------------------------------------------------------------
# Public API methods on exported classes
# (used by library consumers, not by internal code)
# ---------------------------------------------------------------------------

# SchemaResolver — exported from otterapi.codegen
_.resolve_reference
_.get_all_schemas

# TypeRegistry — exported from otterapi and otterapi.codegen
_.register

# ResponseInfo — exported from otterapi.codegen
# is_text is the complement of is_json/is_binary; included for completeness
_.is_text

# TypeInfo dataclass field — exposed to users who iterate TypeRegistry
_.is_generated

# ---------------------------------------------------------------------------
# Test internals
# ---------------------------------------------------------------------------

# MagicMock.return_value setter — the value is consumed by the code under
# test, not read back in the test body.
_.return_value

# Dataclass / pydantic model fields used in test fixtures
_.created_at

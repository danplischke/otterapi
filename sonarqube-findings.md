# SonarQube Community Findings — otterapi

**Scan date:** 2026-06-08  
**SonarQube version:** Community Build 26.6.0.123539  
**Branch:** main (`8dee10f`)

---

## Summary

| Metric | Value |
|---|---|
| Lines of Code | 14,298 |
| Test Coverage | 71.8% |
| Duplicated Lines | 12.6% |
| Bugs | 0 |
| Vulnerabilities | 0 |
| Security Rating | A |
| Reliability Rating | A |
| Technical Debt Ratio | 0.3% |
| **Quality Gate** | **PASSED** |

### Issues by severity

| Severity | Count |
|---|---|
| CRITICAL | 43 |
| MAJOR | 51 |
| MINOR | 12 |
| INFO | 2 |
| **Total** | **129** |

All issues are **Code Smells**. No bugs or vulnerabilities detected.

---

## CRITICAL

### `otterapi/cli.py`

| Line | Message |
|---|---|
| [140](otterapi/cli.py#L140) | Refactor this function to reduce its Cognitive Complexity from 16 to the 15 allowed. |
| [325](otterapi/cli.py#L325) | Refactor this function to reduce its Cognitive Complexity from 45 to the 15 allowed. |

### `otterapi/codegen/client.py`

| Line | Message |
|---|---|
| [168](otterapi/codegen/client.py#L168) | Define a constant instead of duplicating this literal `'HTTP '` 4 times. |

### `otterapi/codegen/codegen.py`

| Line | Message |
|---|---|
| [305](otterapi/codegen/codegen.py#L305) | Refactor this function to reduce its Cognitive Complexity from 17 to the 15 allowed. |
| [526](otterapi/codegen/codegen.py#L526) | Refactor this function to reduce its Cognitive Complexity from 19 to the 15 allowed. |
| [787](otterapi/codegen/codegen.py#L787) | Refactor this function to reduce its Cognitive Complexity from 30 to the 15 allowed. |
| [1307](otterapi/codegen/codegen.py#L1307) | Define a constant instead of duplicating this literal `'.models'` 4 times. |
| [1431](otterapi/codegen/codegen.py#L1431) | Refactor this function to reduce its Cognitive Complexity from 47 to the 15 allowed. |
| [1700](otterapi/codegen/codegen.py#L1700) | Refactor this function to reduce its Cognitive Complexity from 17 to the 15 allowed. |
| [1888](otterapi/codegen/codegen.py#L1888) | Refactor this function to reduce its Cognitive Complexity from 17 to the 15 allowed. |
| [1937](otterapi/codegen/codegen.py#L1937) | Refactor this function to reduce its Cognitive Complexity from 20 to the 15 allowed. |
| [2039](otterapi/codegen/codegen.py#L2039) | Refactor this function to reduce its Cognitive Complexity from 20 to the 15 allowed. |

### `otterapi/codegen/endpoints.py`

| Line | Message |
|---|---|
| [798](otterapi/codegen/endpoints.py#L798) | Refactor this function to reduce its Cognitive Complexity from 37 to the 15 allowed. |
| [802](otterapi/codegen/endpoints.py#L802) | Define a constant instead of duplicating this literal `'pd.DataFrame'` 3 times. |
| [804](otterapi/codegen/endpoints.py#L804) | Define a constant instead of duplicating this literal `'pl.DataFrame'` 3 times. |
| [814](otterapi/codegen/endpoints.py#L814) | Define a constant instead of duplicating this literal `'collections.abc'` 4 times. |
| [917](otterapi/codegen/endpoints.py#L917) | Refactor this function to reduce its Cognitive Complexity from 27 to the 15 allowed. |
| [990](otterapi/codegen/endpoints.py#L990) | Refactor this function to reduce its Cognitive Complexity from 25 to the 15 allowed. |
| [1190](otterapi/codegen/endpoints.py#L1190) | Refactor this function to reduce its Cognitive Complexity from 23 to the 15 allowed. |
| [1322](otterapi/codegen/endpoints.py#L1322) | Refactor this function to reduce its Cognitive Complexity from 24 to the 15 allowed. |
| [1461](otterapi/codegen/endpoints.py#L1461) | Refactor this function to reduce its Cognitive Complexity from 28 to the 15 allowed. |
| [2198](otterapi/codegen/endpoints.py#L2198) | Refactor this function to reduce its Cognitive Complexity from 20 to the 15 allowed. |

### `otterapi/codegen/splitting.py`

| Line | Message |
|---|---|
| [418](otterapi/codegen/splitting.py#L418) | Refactor this function to reduce its Cognitive Complexity from 41 to the 15 allowed. |
| [705](otterapi/codegen/splitting.py#L705) | Refactor this function to reduce its Cognitive Complexity from 51 to the 15 allowed. |
| [1108](otterapi/codegen/splitting.py#L1108) | Define a constant instead of duplicating this literal `'.models'` 4 times. |
| [1174](otterapi/codegen/splitting.py#L1174) | Refactor this function to reduce its Cognitive Complexity from 17 to the 15 allowed. |
| [1223](otterapi/codegen/splitting.py#L1223) | Refactor this function to reduce its Cognitive Complexity from 20 to the 15 allowed. |
| [1327](otterapi/codegen/splitting.py#L1327) | Refactor this function to reduce its Cognitive Complexity from 20 to the 15 allowed. |

### `otterapi/codegen/types.py`

| Line | Message |
|---|---|
| [116](otterapi/codegen/types.py#L116) | Refactor this function to reduce its Cognitive Complexity from 19 to the 15 allowed. |
| [387](otterapi/codegen/types.py#L387) | Refactor this function to reduce its Cognitive Complexity from 54 to the 15 allowed. |
| [470](otterapi/codegen/types.py#L470) | Refactor this function to reduce its Cognitive Complexity from 61 to the 15 allowed. |
| [765](otterapi/codegen/types.py#L765) | Refactor this function to reduce its Cognitive Complexity from 22 to the 15 allowed. |
| [818](otterapi/codegen/types.py#L818) | Refactor this function to reduce its Cognitive Complexity from 36 to the 15 allowed. |

### `otterapi/config.py`

| Line | Message |
|---|---|
| [93](otterapi/config.py#L93) | Define a constant instead of duplicating this literal `'JSON path to extract data from response.'` 3 times. |
| [314](otterapi/config.py#L314) | Refactor this function to reduce its Cognitive Complexity from 18 to the 15 allowed. |
| [490](otterapi/config.py#L490) | Define a constant instead of duplicating this literal `'Per-endpoint configuration overrides.'` 3 times. |
| [495](otterapi/config.py#L495) | Refactor this function to reduce its Cognitive Complexity from 17 to the 15 allowed. |
| [880](otterapi/config.py#L880) | Refactor this function to reduce its Cognitive Complexity from 18 to the 15 allowed. |
| [969](otterapi/config.py#L969) | Define a constant instead of duplicating this literal `'models.py'` 3 times. |
| [976](otterapi/config.py#L976) | Define a constant instead of duplicating this literal `'endpoints.py'` 3 times. |

### `otterapi/openapi/v2/v2.py`

| Line | Message |
|---|---|
| [530](otterapi/openapi/v2/v2.py#L530) | Refactor this function to reduce its Cognitive Complexity from 16 to the 15 allowed. |
| [890](otterapi/openapi/v2/v2.py#L890) | Define a constant instead of duplicating this literal `'application/json'` 5 times. |
| [961](otterapi/openapi/v2/v2.py#L961) | Refactor this function to reduce its Cognitive Complexity from 22 to the 15 allowed. |
| [1184](otterapi/openapi/v2/v2.py#L1184) | Refactor this function to reduce its Cognitive Complexity from 17 to the 15 allowed. |
| [1329](otterapi/openapi/v2/v2.py#L1329) | Refactor this function to reduce its Cognitive Complexity from 44 to the 15 allowed. |
| [1450](otterapi/openapi/v2/v2.py#L1450) | Refactor this function to reduce its Cognitive Complexity from 49 to the 15 allowed. |
| [1635](otterapi/openapi/v2/v2.py#L1635) | Refactor this function to reduce its Cognitive Complexity from 18 to the 15 allowed. |

### `otterapi/openapi/v3/v3.py`

| Line | Message |
|---|---|
| [582](otterapi/openapi/v3/v3.py#L582) | Refactor this function to reduce its Cognitive Complexity from 28 to the 15 allowed. |
| [691](otterapi/openapi/v3/v3.py#L691) | Refactor this function to reduce its Cognitive Complexity from 28 to the 15 allowed. |
| [1085](otterapi/openapi/v3/v3.py#L1085) | Refactor this function to reduce its Cognitive Complexity from 16 to the 15 allowed. |

---

## MAJOR

### `otterapi/codegen/ast_utils.py`

| Line | Message |
|---|---|
| [230](otterapi/codegen/ast_utils.py#L230) | Remove the unused function parameter `reverse_sort`. |

### `otterapi/codegen/client.py`

| Line | Message |
|---|---|
| [62](otterapi/codegen/client.py#L62) | Remove this commented out code. |
| [64](otterapi/codegen/client.py#L64) | Remove this commented out code. |
| [66](otterapi/codegen/client.py#L66) | Remove this commented out code. |
| [68](otterapi/codegen/client.py#L68) | Remove this commented out code. |
| [111](otterapi/codegen/client.py#L111) | Remove this commented out code. |
| [113](otterapi/codegen/client.py#L113) | Remove this commented out code. |
| [115](otterapi/codegen/client.py#L115) | Remove this commented out code. |
| [120](otterapi/codegen/client.py#L120) | Remove this commented out code. |
| [128](otterapi/codegen/client.py#L128) | Remove this commented out code. |
| [138](otterapi/codegen/client.py#L138) | Remove this commented out code. |
| [148](otterapi/codegen/client.py#L148) | Remove this commented out code. |
| [163](otterapi/codegen/client.py#L163) | Remove this commented out code. |
| [184](otterapi/codegen/client.py#L184) | Remove this commented out code. |
| [386](otterapi/codegen/client.py#L386) | Remove this commented out code. |
| [747](otterapi/codegen/client.py#L747) | Remove this commented out code. |
| [752](otterapi/codegen/client.py#L752) | Remove this commented out code. |
| [757](otterapi/codegen/client.py#L757) | Remove this commented out code. |
| [765](otterapi/codegen/client.py#L765) | Remove this commented out code. |
| [770](otterapi/codegen/client.py#L770) | Remove this commented out code. |
| [919](otterapi/codegen/client.py#L919) | Remove this commented out code. |
| [924](otterapi/codegen/client.py#L924) | Remove this commented out code. |
| [945](otterapi/codegen/client.py#L945) | Remove this commented out code. |
| [954](otterapi/codegen/client.py#L954) | Remove this commented out code. |
| [1037](otterapi/codegen/client.py#L1037) | Remove this commented out code. |
| [1107](otterapi/codegen/client.py#L1107) | Remove this commented out code. |
| [1119](otterapi/codegen/client.py#L1119) | Remove this commented out code. |
| [1249](otterapi/codegen/client.py#L1249) | Remove this commented out code. |
| [1332](otterapi/codegen/client.py#L1332) | Remove this commented out code. |
| [1392](otterapi/codegen/client.py#L1392) | Remove the unused function parameter `response_infos`. |
| [1397](otterapi/codegen/client.py#L1397) | Remove this commented out code. |
| [1400](otterapi/codegen/client.py#L1400) | Remove this commented out code. |
| [1414](otterapi/codegen/client.py#L1414) | Remove this commented out code. |
| [1423](otterapi/codegen/client.py#L1423) | Remove this commented out code. |

### `otterapi/codegen/codegen.py`

| Line | Message |
|---|---|
| [788](otterapi/codegen/codegen.py#L788) | Remove the unused function parameter `baseurl`. |
| [1284](otterapi/codegen/codegen.py#L1284) | Remove the unused function parameter `models_file`. |

### `otterapi/codegen/endpoints.py`

| Line | Message |
|---|---|
| [1119](otterapi/codegen/endpoints.py#L1119) | Remove this commented out code. |
| [1125](otterapi/codegen/endpoints.py#L1125) | Remove this commented out code. |
| [1746](otterapi/codegen/endpoints.py#L1746) | Remove the unused function parameter `is_async`. |

### `otterapi/codegen/schema.py`

| Line | Message |
|---|---|
| [393](otterapi/codegen/schema.py#L393) | Replace chained `startswith` calls with a single call using a tuple argument. |
| [400](otterapi/codegen/schema.py#L400) | Replace chained `startswith` calls with a single call using a tuple argument. |

### `otterapi/codegen/splitting.py`

| Line | Message |
|---|---|
| [709](otterapi/codegen/splitting.py#L709) | Remove the unused function parameter `base_url`. |
| [711](otterapi/codegen/splitting.py#L711) | Remove the unused function parameter `module_path`. |
| [1526](otterapi/codegen/splitting.py#L1526) | Remove the unused function parameter `tree`. |

### `otterapi/codegen/types.py`

| Line | Message |
|---|---|
| [61](otterapi/codegen/types.py#L61) | Rename field `type` (conflicts with built-in). |
| [125](otterapi/codegen/types.py#L125) | Rename field `type` (conflicts with built-in). |
| [758](otterapi/codegen/types.py#L758) | Return a value of type `str` instead of `AnnAssign` or update function `_create_pydantic_field` type hint. |
| [976](otterapi/codegen/types.py#L976) | Remove the unused function parameter `name`. |

### `otterapi/codegen/utils.py`

| Line | Message |
|---|---|
| [159](otterapi/codegen/utils.py#L159) | Replace spaces with quantifier `{4}`. |

### `otterapi/openapi/v2/v2.py`

| Line | Message |
|---|---|
| [353](otterapi/openapi/v2/v2.py#L353) | Use a union type expression for this type hint. |
| [356](otterapi/openapi/v2/v2.py#L356) | Use a union type expression for this type hint. |

### `otterapi/openapi/v3/v3.py`

| Line | Message |
|---|---|
| [180](otterapi/openapi/v3/v3.py#L180) | Assign to `style` a value of type `Optional[Style1]` instead of `str` or update its type hint. |
| [197](otterapi/openapi/v3/v3.py#L197) | Assign to `style` a value of type `Optional[Style2]` instead of `str` or update its type hint. |
| [210](otterapi/openapi/v3/v3.py#L210) | Assign to `style` a value of type `Optional[Style3]` instead of `str` or update its type hint. |
| [223](otterapi/openapi/v3/v3.py#L223) | Assign to `style` a value of type `Optional[Style4]` instead of `str` or update its type hint. |
| [1452](otterapi/openapi/v3/v3.py#L1452) | Assign to `style` a value of type `Optional[Style]` instead of `str` or update its type hint. |

### `otterapi/openapi/v3_1/v3_1.py`

| Line | Message |
|---|---|
| [174](otterapi/openapi/v3_1/v3_1.py#L174) | Assign to `style` a value of type `Optional[Style1]` instead of `str` or update its type hint. |
| [197](otterapi/openapi/v3_1/v3_1.py#L197) | Assign to `style` a value of type `Optional[Style2]` instead of `str` or update its type hint. |
| [216](otterapi/openapi/v3_1/v3_1.py#L216) | Assign to `style` a value of type `Optional[Style3]` instead of `str` or update its type hint. |
| [235](otterapi/openapi/v3_1/v3_1.py#L235) | Assign to `style` a value of type `Optional[Style4]` instead of `str` or update its type hint. |
| [617](otterapi/openapi/v3_1/v3_1.py#L617) | Assign to `style` a value of type `Optional[Style]` instead of `str` or update its type hint. |

### `otterapi/openapi/v3_2/v3_2.py`

| Line | Message |
|---|---|
| [169](otterapi/openapi/v3_2/v3_2.py#L169) | Assign to `style` a value of type `Optional[Style1]` instead of `str` or update its type hint. |
| [192](otterapi/openapi/v3_2/v3_2.py#L192) | Assign to `style` a value of type `Optional[Style2]` instead of `str` or update its type hint. |
| [211](otterapi/openapi/v3_2/v3_2.py#L211) | Assign to `style` a value of type `Optional[Style3]` instead of `str` or update its type hint. |
| [230](otterapi/openapi/v3_2/v3_2.py#L230) | Assign to `style` a value of type `Optional[Style4]` instead of `str` or update its type hint. |
| [636](otterapi/openapi/v3_2/v3_2.py#L636) | Assign to `style` a value of type `Optional[Style]` instead of `str` or update its type hint. |

---

## MINOR

### `otterapi/codegen/codegen.py`

| Line | Message |
|---|---|
| [1016](otterapi/codegen/codegen.py#L1016) | Remove this unneeded `pass`. |

### `otterapi/codegen/types.py`

| Line | Message |
|---|---|
| [78](otterapi/codegen/types.py#L78) | Use `set.update()` instead of a for-loop with `add()`. |
| [713](otterapi/codegen/types.py#L713) | Replace this constructor call with a literal. |

### `otterapi/codegen/utils.py`

| Line | Message |
|---|---|
| [56](otterapi/codegen/utils.py#L56) | Use concise character class syntax `\W` instead of `[^A-Za-z0-9_]`. |
| [112](otterapi/codegen/utils.py#L112) | Use concise character class syntax `\W` instead of `[^A-Za-z0-9_]`. |

### `otterapi/openapi/v2/v2.py`

| Line | Message |
|---|---|
| [224](otterapi/openapi/v2/v2.py#L224) | Replace the unused local variable `plural` with `_`. |

### `otterapi/openapi/v3/v3.py`

| Line | Message |
|---|---|
| [492](otterapi/openapi/v3/v3.py#L492) | Replace this comprehension with passing the iterable to the dict constructor call. |
| [737](otterapi/openapi/v3/v3.py#L737) | Rename this local variable `allOf` to match the regular expression `^[_a-z][a-z0-9_]*$`. |
| [742](otterapi/openapi/v3/v3.py#L742) | Rename this local variable `oneOf` to match the regular expression `^[_a-z][a-z0-9_]*$`. |
| [747](otterapi/openapi/v3/v3.py#L747) | Rename this local variable `anyOf` to match the regular expression `^[_a-z][a-z0-9_]*$`. |
| [1323](otterapi/openapi/v3/v3.py#L1323) | Replace this comprehension with passing the iterable to the dict constructor call. |

---

## INFO

### `otterapi/codegen/codegen.py`

| Line | Message |
|---|---|
| [722](otterapi/codegen/codegen.py#L722) | Complete the task associated to this `TODO` comment. |

### `otterapi/codegen/types.py`

| Line | Message |
|---|---|
| [1121](otterapi/codegen/types.py#L1121) | Complete the task associated to this `TODO` comment. |

---

## Patterns worth addressing

**Cognitive Complexity** (35 of 43 CRITICAL issues) — the codegen and OpenAPI parsing layers have grown large deeply-nested functions. The worst offenders:

| File | Line | Complexity |
|---|---|---|
| `otterapi/codegen/types.py` | [470](otterapi/codegen/types.py#L470) | 61 |
| `otterapi/codegen/splitting.py` | [705](otterapi/codegen/splitting.py#L705) | 51 |
| `otterapi/openapi/v2/v2.py` | [1450](otterapi/openapi/v2/v2.py#L1450) | 49 |
| `otterapi/codegen/codegen.py` | [1431](otterapi/codegen/codegen.py#L1431) | 47 |
| `otterapi/codegen/splitting.py` | [418](otterapi/codegen/splitting.py#L418) | 41 |
| `otterapi/codegen/types.py` | [387](otterapi/codegen/types.py#L387) | 54 |

**Commented-out code** (32 MAJOR issues) — concentrated almost entirely in `client.py`. Git history serves the same purpose.

**Unused parameters** (8 MAJOR issues) — across `client.py`, `codegen.py`, `endpoints.py`, `splitting.py`, and `types.py`. Dead API surface that makes signatures misleading.

**`style` type mismatches** (10 MAJOR issues) — repeated across `v3.py`, `v3_1.py`, and `v3_2.py`; the field is typed as `str` but assigned Optional enum values.

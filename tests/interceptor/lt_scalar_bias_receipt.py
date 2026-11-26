"""Independent exact raw-image audit for the native-only Lt scalar/bias matrix."""

from functools import lru_cache
import json
import struct

ELEMENTS = dict(a=2336, b=3872, c=6944, d=7712, bias=96, scalars=8)
FIELDS = {"id", "format", "device_scalars", "bias", "in_place", "zero_beta",
          "algorithm", "pointer_mask", "epilogue_mask", "mismatches", "guard_mismatches",
          "inputs_unchanged", "output_bytes"}
FLAGS = ("full_goal_complete", "full_device_qualified", "physical_priority_qualified",
         "performance_qualified")
FAILURE = "native Lt full-value oracle failed after complete cleanup"


def require(value, message):
    if not value:
        raise ValueError(message)


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate diagnostic JSON field")
        value[key] = item
    return value


def planned_cases():
    first = (1, True, True, False, False)
    return [first] + [(format_, device, bias, alias, zero)
                      for format_ in range(3) for device in (False, True)
                      for bias in (False, True) for alias in (False, True)
                      for zero in (False, True)
                      if (format_, device, bias, alias, zero) != first]


@lru_cache(maxsize=6)
def initial_images(format_, zero):
    result = {}
    for name, count in ELEMENTS.items():
        half = format_ != 0 if name in ("a", "b") else format_ == 1
        if name == "scalars":
            half = False
            values = [2., 0., 0., 0., 0. if zero else .5, 0., 0., 0.]
        else:
            values = [(index % 9 - 4) / 8 for index in range(count)]
        result[name] = struct.pack("<" + ("e" if half else "f") * count, *values)
    return result


@lru_cache(maxsize=1)
def products():
    return [sum(((row + inner * 72) % 9 - 4) * ((inner + column * 40) % 9 - 4)
                for inner in range(32)) for column in range(96) for row in range(64)]


def expected_image(case):
    half = case["format"] == 1
    code, size = ("<e", 2) if half else ("<f", 4)
    output = "c" if case["in_place"] else "d"
    leading = 72 if case["in_place"] else 80
    result = bytearray(initial_images(case["format"], case["zero_beta"])[output])
    for column in range(96):
        for row in range(64):
            numerator = 2 * products()[column * 64 + row]
            if not case["zero_beta"]:
                numerator += 4 * ((row + column * 72) % 9 - 4)
            if case["bias"]:
                numerator += 8 * (row % 9 - 4)
            struct.pack_into(code, result, size * (row + column * leading), numerator / 64)
    return bytes(result)


def image_names(case_id):
    return {f"{case_id}-{name}-{phase}.bin" for name in ELEMENTS for phase in ("before", "after")} | {
        f"{case_id}-expected.bin"}


def audit_case(case, images):
    require(set(case) == FIELDS, "unexpected diagnostic case fields")
    case_id = case["id"]
    for name in ("id", "format", "algorithm", "pointer_mask", "epilogue_mask", "mismatches",
                 "guard_mismatches", "output_bytes"):
        require(type(case[name]) is int and 0 <= case[name] < 2**32,
                "bounded nonnegative integer diagnostic field required")
    for name in ("device_scalars", "bias", "in_place", "zero_beta", "inputs_unchanged"):
        require(type(case[name]) is bool, "exact diagnostic boolean required")
    require(case["format"] in (0, 1, 2), "unknown diagnostic format")
    initial = initial_images(case["format"], case["zero_beta"])
    output = "c" if case["in_place"] else "d"
    expected = expected_image(case)
    require(case["output_bytes"] == len(expected), "incorrect full output span")
    def read(name, size):
        path = images / name
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == size,
                "missing, aliased or truncated full diagnostic image")
        return path.read_bytes()
    require(read(f"{case_id}-expected.bin", len(expected)) == expected,
            "SDK expected-image file differs from independent integer oracle")
    actual = {}
    for name, before in initial.items():
        require(read(f"{case_id}-{name}-before.bin", len(before)) == before,
                "diagnostic initial input differs from fixed independent pattern")
        actual[name] = read(f"{case_id}-{name}-after.bin", len(before))
    inputs_unchanged = all(actual[name] == initial[name] for name in initial if name != output)
    leading, size = (72 if case["in_place"] else 80), (2 if case["format"] == 1 else 4)
    mismatches = guards = missing_product_matches = discriminating_outputs = 0
    for index in range(len(expected) // size):
        observed = actual[output][index * size:(index + 1) * size]
        wanted = expected[index * size:(index + 1) * size]
        column, row = divmod(index, leading)
        if column < 96 and row < 64:
            mismatches += observed != wanted
            base = 0 if case["zero_beta"] else 4 * ((row + column * 72) % 9 - 4)
            if case["bias"]:
                base += 8 * (row % 9 - 4)
            no_product = struct.pack("<e" if size == 2 else "<f", base / 64)
            if no_product != wanted:
                discriminating_outputs += 1
                missing_product_matches += observed == no_product
        else:
            guards += observed != wanted
    require((mismatches, guards, inputs_unchanged) ==
            (case["mismatches"], case["guard_mismatches"], case["inputs_unchanged"]),
            "reported mismatch/input counts differ from complete raw images")
    return dict(id=case_id, mismatches=mismatches, guard_mismatches=guards,
                inputs_unchanged=inputs_unchanged, checked_output_elements=len(expected) // size,
                discriminating_outputs=discriminating_outputs,
                missing_product_matches=missing_product_matches,
                advertised_pointer_mode=bool(case["pointer_mask"] & (2 if case["device_scalars"] else 1)),
                advertised_epilogue=bool(case["epilogue_mask"] & (4 if case["bias"] else 1)))


def validate(receipt_path, images):
    require(not images.is_symlink() and images.is_dir(), "exact diagnostic image directory required")
    require(not receipt_path.is_symlink() and receipt_path.is_file() and
            receipt_path.stat().st_size <= 1024 * 1024, "aliased or oversized diagnostic receipt")
    lines = receipt_path.read_text().splitlines()
    require(1 <= len(lines) <= 2, "exact complete diagnostic receipt required")
    value = json.loads(lines[0], object_pairs_hook=unique_object,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    require(type(value) is dict and set(value) ==
            {"status", "cleanup", "diagnostic_only", "lt_version", "blas_version", "cases", *FLAGS},
            "unexpected diagnostic receipt fields")
    require(value["cleanup"] is True and value["diagnostic_only"] is True and
            all(value[key] is False for key in FLAGS), "diagnostic cannot qualify full objective")
    require(type(value["lt_version"]) is int and value["lt_version"] > 0 and
            type(value["blas_version"]) is list and len(value["blas_version"]) == 3 and
            all(type(item) is int and item >= 0 for item in value["blas_version"]), "invalid SDK version")
    require(type(value["cases"]) is list and len(value["cases"]) == 48, "complete fixed48-case matrix required")
    expected_names = set().union(*(image_names(index) for index in range(48)))
    require({path.name for path in images.iterdir()} == expected_names, "missing or extra diagnostic images")
    audited = []
    for index, (case, planned) in enumerate(zip(value["cases"], planned_cases(), strict=True)):
        require(type(case) is dict and case.get("id") == index and
                tuple(case.get(key) for key in ("format", "device_scalars", "bias", "in_place", "zero_beta")) == planned,
                "first isolated case or fixed matrix order changed")
        audited.append(audit_case(case, images))
    passed = all(not case["mismatches"] and not case["guard_mismatches"] and case["inputs_unchanged"]
                 for case in audited)
    require(value["status"] == ("passed" if passed else "numerical_failure") and
            lines[1:] == ([] if passed else [FAILURE]), "wrong numerical result or missing cleanup-complete failure")
    return dict(status="native_matrix_numerics_passed" if passed else "native_matrix_numerics_failed",
                numerical_passed=passed, cleanup=True, case_audits=audited, receipt=value,
                saved_images_independently_reloaded=True, full_goal_complete=False,
                full_device_qualified=False, physical_priority_qualified=False, performance_qualified=False)

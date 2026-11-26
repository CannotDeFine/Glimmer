#!/usr/bin/env python3
"""Export the pinned full BERT QA inference oracle to a local FP32 ONNX model."""

import argparse
import json
import pathlib

from bert_contract import SEED, configure_cache, sha256, validate_assets, write_json
from bert_workload import make_batch


def export(assets, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "seed": SEED, "dtype": "float32", "opset": 17}
    try:
        report["assets"] = validate_assets(assets)
        configure_cache()
        import torch
        import transformers
        import onnx
        from transformers import BertForQuestionAnswering, BertTokenizerFast

        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tokenizer = BertTokenizerFast.from_pretrained(assets, local_files_only=True)
        model = BertForQuestionAnswering.from_pretrained(
            assets, local_files_only=True, use_safetensors=True, attn_implementation="eager").eval().cuda()
        inputs, features = make_batch(torch, tokenizer, assets, "inference")
        device_inputs = {name: value.cuda() for name, value in inputs.items()}
        with torch.inference_mode():
            reference = model(**device_inputs)
        for value in (reference.start_logits, reference.end_logits):
            if not torch.isfinite(value).all().item():
                raise ValueError("nonfinite export oracle")
        path = output / "bert.onnx"
        torch.onnx.export(model, tuple(device_inputs.values()), str(path),
                          input_names=list(inputs), output_names=["start_logits", "end_logits"],
                          opset_version=17, dynamo=False, external_data=False)
        onnx.checker.check_model(str(path))
        report.update(status="ok", inputs=features, model_sha256=sha256(path),
                      input_names=list(inputs), output_names=["start_logits", "end_logits"],
                      reference={"start_logits": reference.start_logits.cpu().tolist(),
                                 "end_logits": reference.end_logits.cpu().tolist()},
                      environment={"torch": torch.__version__, "cuda": torch.version.cuda,
                                   "transformers": transformers.__version__, "onnx": onnx.__version__},
                      source_sha256={name: sha256(pathlib.Path(__file__).with_name(name))
                                     for name in ("export_bert.py", "bert_workload.py", "bert_contract.py")})
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        write_json(output / "export.json", report)
    print(json.dumps({"status": report["status"], "error": report.get("error")}))
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    return export(arguments.assets, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())

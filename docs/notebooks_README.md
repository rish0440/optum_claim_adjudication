# Notebooks

`qwen25_vl_colab.ipynb` runs the active `trainer.sft_qlora_invoice_extraction` CORD workflow.
Build its upload bundle with `python -m scripts.prepare_colab_dataset`.
Only 161 training and 20 validation receipts are included; held-out test files
and original Optum evaluation invoices are excluded. The notebook creates its
own temporary Colab training environment. No notebook is required for the demo.

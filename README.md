# MiniLM in Wasm 
This is a demo of running a port of the [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model to dfdx that can run in WASM on a browser.
The minilm model is a good model for semantic retrieval tasks, and is ideally suited for client side implementation because it is quite lite weight.
![screendhot of the minilm model running in a browser showing "Asian food" and "Thai curry" having a similarity of about 0.59](preview.png)

In order to run it locally you'll need to download some model files from the huggingface hub. This can by done from the `python` directory by running
Dependencies for this
* [poetry](https://python-poetry.org/)
* [trunk](https://trunkrs.dev/)
* [rust](https://rustup.rs/) (obviously, but also it's nightly)

```sh
poetry install
poetry run python minilm/load_model.py
```

This will create a `sites/model.safetensors` in the top level `sites` directory. 
The install step will create a virtual environment, which can be determined by running
```sh
poetry run which python
```
The location of this virtual environment is useful for testing.

Now to start the site up, go to the `site` directory and run 
```sh
trunk serve --open --release
```

To run the test you will need to run 
```sh
PYO3_PYTHON=$VIRTUAL_ENV_PATH/bin/python PYTHONPATH=$VIRTUAL_ENV_PATH/lib/python3.11/site-packages cargo test -F pyo3 embeddings
```
where `VIRTUAL_ENV_PATH` ithe path of the the poetry generated virtual environment (before `/bin/python`).



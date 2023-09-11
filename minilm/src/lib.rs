#![feature(generic_const_exprs)]
#![feature(inline_const)]
use dfdx::shapes::Rank2;
mod embeddings;
mod transformer_layer;

pub type DfdxLinear = dfdx::nn::modules::Linear<5, 7, f32, Cpu>;
pub type DfdxTensor = dfdx::tensor::Tensor<Rank2<16, 5>, f32, Cpu>;

pub struct Test(DfdxLinear, DfdxTensor);

pub use dfdx::tensor::Cpu;
pub use dfdx::tensor::CpuError;
pub use tokenizers::Tokenizer;
pub use transformer_layer::MiniLM;

#[cfg(test)]
mod tests {

    use std::{io::Read, path::Path};

    use dfdx::{
        shapes::{Dtype, Shape},
        tensor::{safetensors::SafeDtype, Tensor},
        tensor_ops::Device,
    };
    use safetensors::SafeTensors;

    pub fn load_tensor<P: AsRef<Path>, S: Shape, E: Dtype + SafeDtype, D: Device<E>>(
        path: P,
        name: &str,
        tensor: &mut Tensor<S, E, D>,
    ) {
        let path_ref = path.as_ref();
        let mut f = std::fs::File::open(path_ref)
            .unwrap_or_else(|_| panic!("unable to open file {path_ref:#?} should be there"));
        let mut data = vec![];
        f.read_to_end(&mut data);
        println!("data length {}", data.len());
        let tensors = SafeTensors::deserialize(data.as_mut()).unwrap_or_else(|err| {
            panic!("Unable be able to read safe_tensor from file, {path_ref:#?}: error: {err}")
        });

        tensor.load_safetensor(&tensors, name).unwrap_or_else(|err| {
            panic!("Unable to find tensor name {name} in safetensors file {path_ref:#?}: error: {err:#?}")
        })
    }

    #[cfg(feature = "pyo3")]
    #[test]
    fn test_embeddings() {
        use dfdx::{
            prelude::{BuildModule, LoadFromSafetensors, Module},
            shapes::{Const, HasShape},
            tensor::{Cpu, TensorFromVec, ZerosTensor},
        };
        use pyo3::Python;
        use tokenizers::Tokenizer;

        use crate::embeddings::BertEmbeddings;

        pyo3::prepare_freethreaded_python();

        let tokenizer = Tokenizer::from_file("../python/models/tokenizer/tokenizer.json")
            .expect("Failed to open tokenizer");

        let encoded = tokenizer
            .encode("Hello my name is Bort", true)
            .expect("Failed to tokenize \"Hello my name is Bort\"");

        let dev = Cpu::default();
        let ids: Vec<_> = encoded.get_ids().iter().map(|i| *i as usize).collect();
        let length = ids.len();
        let ids: Tensor<(usize,), usize, Cpu> = dev.tensor_from_vec(ids, (length,));

        let mut embeddings: BertEmbeddings<384, 30522, 512, f32, Cpu> = BertEmbeddings::build(&dev);

        // We can load a buffer just with
        //
        // let mut tensors = SafeTensors::deserialize(&buffer)?;
        // Self::iter_tensors(&mut RecursiveWalker {
        //     m: (self, String::new()),
        //     f: &mut tensors,
        // })?;

        // gpt2.save_safetensors("test_data/gpt2_test.safetensors");
        embeddings
            .load_safetensors("../python/models/embeddings.safetensors")
            .expect("Failed to load safetensors");

        let output = embeddings
            .try_forward(ids)
            .expect("failed to call apply embeddings to ids");

        Python::with_gil(|py| {
            py.run(
                r#"
import torch as to
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

output = model[0].auto_model.embeddings(to.LongTensor([model.tokenizer.encode("Hello my name is Bort")])).squeeze(0)
save_file({"output": output}, "../python/models/output.safetensors")
                   "#,
                None,
                None,
            ).expect("Failed to run python");
        });

        let mut output_should: Tensor<(usize, Const<384>), f32, Cpu> =
            dev.zeros_like(output.shape());
        load_tensor(
            "../python/models/output.safetensors",
            "output",
            &mut output_should,
        );

        let [a, b] = output.shape().concrete();
        let diff = (output - output_should).abs().as_vec().iter().sum::<f32>();
        println!("DIFF: {}", diff);
        assert!(diff / (a * b) as f32 <= 1.0e-7)
    }
}

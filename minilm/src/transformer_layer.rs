use std::marker::PhantomData;

use dfdx::{
    prelude::{
        modules::{LayerNorm1D, Linear},
        AccurateGeLU, BuildModule, Dropout, Module, RecursiveWalker, Repeated, Softmax, Tanh,
        TensorCollection, TensorOptions,
    },
    shapes::{Axes3, Const, Dtype, HasShape, Rank1, Shape},
    tensor::{Cpu, CpuError, Tensor, TensorFromVec, ZerosTensor},
    tensor_ops::{
        BroadcastTo, Device, MeanTo, PermuteTo, RealizeTo, ReshapeTo, SumTo, TryAdd, TryDiv,
        TryMatMul,
    },
};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

use crate::embeddings::BertEmbeddings;

pub struct TransformerLayer<const DIM: usize, E: Dtype, D: Device<E>> {
    _x: PhantomData<[E; DIM]>,
    _y: PhantomData<D>,
}

pub struct SelfAttention<const HEAD_DIM: usize, const N_HEADS: usize, E: Dtype, D: Device<E>>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
{
    key: Linear<{ HEAD_DIM * N_HEADS }, { HEAD_DIM * N_HEADS }, E, D>,
    query: Linear<{ HEAD_DIM * N_HEADS }, { HEAD_DIM * N_HEADS }, E, D>,
    value: Linear<{ HEAD_DIM * N_HEADS }, { HEAD_DIM * N_HEADS }, E, D>,
    dropout: Dropout,
    softmax: Softmax,
}

impl<
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D> for SelfAttention<HEAD_DIM, N_HEADS, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
{
    type To<E2: Dtype, D2: Device<E2>> = SelfAttention<HEAD_DIM, N_HEADS, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("key", |s| &s.key, |s| &mut s.key),
                Self::module("query", |s| &s.query, |s| &mut s.query),
                Self::module("value", |s| &s.value, |s| &mut s.value),
            ),
            |(key, query, value)| SelfAttention {
                key,
                query,
                value,
                dropout: Dropout { p: 0.0 },
                softmax: Softmax,
            },
        )
    }
}

impl<const HEAD_DIM: usize, const N_HEADS: usize, E, D>
    Module<Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>>
    for SelfAttention<HEAD_DIM, N_HEADS, E, D>
where
    E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
{
    type Output = Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let [length, _] = input.shape().concrete();
        let key = self
            .key
            .try_forward(input.clone())?
            .try_reshape_like::<(usize, Const<N_HEADS>, Const<HEAD_DIM>)>(&(length, Const, Const))?
            .try_permute::<_, Axes3<1, 2, 0>>()?;
        let query = self
            .query
            .try_forward(input.clone())?
            .try_reshape_like::<(usize, Const<N_HEADS>, Const<HEAD_DIM>)>(&(length, Const, Const))?
            .try_permute::<_, Axes3<1, 0, 2>>()?;
        let value = self
            .value
            .try_forward(input)?
            .try_reshape_like::<(usize, Const<N_HEADS>, Const<HEAD_DIM>)>(&(length, Const, Const))?
            .try_permute::<_, Axes3<1, 0, 2>>()?;
        let attention_score: Tensor<(Const<N_HEADS>, usize, usize), E, D> =
            query.try_matmul(key)?;

        let attention_score = attention_score.try_div(
            E::from_usize(HEAD_DIM)
                .expect(&format!("failed to cast {HEAD_DIM} as dtype"))
                .sqrt(),
        )?;
        let attention_probs = self.softmax.try_forward(attention_score)?;
        let result: Tensor<(Const<N_HEADS>, usize, Const<HEAD_DIM>), E, D> =
            attention_probs.try_matmul(value)?;
        result
            .try_permute::<_, Axes3<1, 0, 2>>()?
            .try_reshape_like(&(length, Const))
    }
}

struct SelfOutput<const DIM: usize, E: Dtype, D: Device<E>> {
    dense: Linear<DIM, DIM, E, D>,
    norm: LayerNorm1D<DIM, E, D>,
    dropout: Dropout,
}

impl<
        const DIM: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D> for SelfOutput<DIM, E, D>
{
    type To<E2: Dtype, D2: Device<E2>> = SelfOutput<DIM, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("dense", |s| &s.dense, |s| &mut s.dense),
                Self::tensor(
                    "LayerNorm.weight",
                    |s| &s.norm.gamma,
                    |s| &mut s.norm.gamma,
                    TensorOptions::reset_to_ones(),
                ),
                Self::tensor(
                    "LayerNorm.bias",
                    |s| &s.norm.beta,
                    |s| &mut s.norm.beta,
                    TensorOptions::reset_to_ones(),
                ),
            ),
            |(output, gamma, beta)| SelfOutput {
                dense: output,
                norm: LayerNorm1D {
                    gamma,
                    beta,
                    epsilon: 1.0e-11,
                },
                dropout: Dropout { p: 0.0 },
            },
        )
    }
}

impl<
        const DIM: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    >
    Module<(
        Tensor<(usize, Const<DIM>), E, D>,
        Tensor<(usize, Const<DIM>), E, D>,
    )> for SelfOutput<DIM, E, D>
{
    type Output = Tensor<(usize, Const<DIM>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        (input, hidden_state): (
            Tensor<(usize, Const<DIM>), E, D>,
            Tensor<(usize, Const<DIM>), E, D>,
        ),
    ) -> Result<Self::Output, Self::Error> {
        let linear_out = self.dense.try_forward(input)?;
        let dropped = self.dropout.try_forward(linear_out)?;
        self.norm.try_forward(dropped.try_add(hidden_state)?)
    }
}

struct Attention<const HEAD_DIM: usize, const N_HEADS: usize, E: Dtype, D: Device<E>>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
{
    self_attn: SelfAttention<HEAD_DIM, N_HEADS, E, D>,
    output: SelfOutput<{ HEAD_DIM * N_HEADS }, E, D>,
}

impl<
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D> for Attention<HEAD_DIM, N_HEADS, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
{
    type To<E2: Dtype, D2: Device<E2>> = Attention<HEAD_DIM, N_HEADS, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("self", |s| &s.self_attn, |s| &mut s.self_attn),
                Self::module("output", |s| &s.output, |s| &mut s.output),
            ),
            |(self_attn, output)| Attention { self_attn, output },
        )
    }
}

impl<
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > Module<Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>>
    for Attention<HEAD_DIM, N_HEADS, E, D>
{
    type Output = Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let attn_out = self.self_attn.try_forward(input.clone())?;
        self.output.try_forward((attn_out, input))
    }
}

struct Intermediate<const DIM: usize, E: Dtype, D: Device<E>>
where
    [(); DIM * 4]: Sized,
{
    dense: Linear<DIM, { DIM * 4 }, E, D>,
    activation: AccurateGeLU,
}

impl<
        const DIM: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D> for Intermediate<DIM, E, D>
where
    [(); DIM * 4]: Sized,
{
    type To<E2: Dtype, D2: Device<E2>> = Intermediate<DIM, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            Self::module("dense", |s| &s.dense, |s| &mut s.dense),
            |dense| Intermediate {
                dense,
                activation: AccurateGeLU,
            },
        )
    }
}

impl<
        const DIM: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > Module<Tensor<(usize, Const<DIM>), E, D>> for Intermediate<DIM, E, D>
where
    [(); DIM * 4]: Sized,
{
    type Output = Tensor<(usize, Const<{ DIM * 4 }>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<DIM>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        self.activation.try_forward(self.dense.try_forward(input)?)
    }
}

struct Output<const DIM: usize, E: Dtype, D: Device<E>>
where
    [(); DIM * 4]: Sized,
{
    dense: Linear<{ DIM * 4 }, DIM, E, D>,
    norm: LayerNorm1D<DIM, E, D>,
    dropout: Dropout,
}

impl<
        const DIM: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D> for Output<DIM, E, D>
where
    [(); DIM * 4]: Sized,
{
    type To<E2: Dtype, D2: Device<E2>> = Output<DIM, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("dense", |s| &s.dense, |s| &mut s.dense),
                Self::tensor(
                    "LayerNorm.weight",
                    |s| &s.norm.gamma,
                    |s| &mut s.norm.gamma,
                    TensorOptions::reset_to_ones(),
                ),
                Self::tensor(
                    "LayerNorm.bias",
                    |s| &s.norm.beta,
                    |s| &mut s.norm.beta,
                    TensorOptions::reset_to_ones(),
                ),
            ),
            |(dense, gamma, beta)| Output {
                dense,
                norm: LayerNorm1D {
                    gamma,
                    beta,
                    epsilon: 1.0e-12,
                },
                dropout: Dropout { p: 0.0 },
            },
        )
    }
}

impl<
        const DIM: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    >
    Module<(
        Tensor<(usize, Const<{ DIM * 4 }>), E, D>,
        Tensor<(usize, Const<DIM>), E, D>,
    )> for Output<DIM, E, D>
where
    [(); DIM * 4]: Sized,
{
    type Output = Tensor<(usize, Const<DIM>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        (input, hidden): (
            Tensor<(usize, Const<{ DIM * 4 }>), E, D>,
            Tensor<(usize, Const<DIM>), E, D>,
        ),
    ) -> Result<Self::Output, Self::Error> {
        self.norm.try_forward(
            self.dropout
                .try_forward(self.dense.try_forward(input)?)?
                .try_add(hidden)?,
        )
    }
}

struct Layer<const HEAD_DIM: usize, const N_HEADS: usize, E: Dtype, D: Device<E>>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); HEAD_DIM * N_HEADS * 4]: Sized,
{
    attention: Attention<HEAD_DIM, N_HEADS, E, D>,
    intermediate: Intermediate<{ HEAD_DIM * N_HEADS }, E, D>,
    output: Output<{ HEAD_DIM * N_HEADS }, E, D>,
}

impl<
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D> for Layer<HEAD_DIM, N_HEADS, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); HEAD_DIM * N_HEADS * 4]: Sized,
{
    type To<E2: Dtype, D2: Device<E2>> = Layer<HEAD_DIM, N_HEADS, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("attention", |s| &s.attention, |s| &mut s.attention),
                Self::module("intermediate", |s| &s.intermediate, |s| &mut s.intermediate),
                Self::module("output", |s| &s.output, |s| &mut s.output),
            ),
            |(attention, intermediate, output)| Layer {
                attention,
                intermediate,
                output,
            },
        )
    }
}

impl<
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > Module<Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>>
    for Layer<HEAD_DIM, N_HEADS, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); HEAD_DIM * N_HEADS * 4]: Sized,
{
    type Output = Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>;
    type Error = D::Err;
    fn try_forward(
        &self,
        input: Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let hidden = self.attention.try_forward(input)?;
        let intermediate_out = self.intermediate.try_forward(hidden.clone())?;
        self.output.try_forward((intermediate_out, hidden))
    }
}

struct Encoder<
    const HEAD_DIM: usize,
    const N_HEADS: usize,
    const N_LAYERS: usize,
    E: Dtype,
    D: Device<E>,
> where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); HEAD_DIM * N_HEADS * 4]: Sized,
{
    layer: Repeated<Layer<HEAD_DIM, N_HEADS, E, D>, N_LAYERS>,
}

impl<
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        const N_LAYERS: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D> for Encoder<HEAD_DIM, N_HEADS, N_LAYERS, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); HEAD_DIM * N_HEADS * 4]: Sized,
{
    type To<E2: Dtype, D2: Device<E2>> = Encoder<HEAD_DIM, N_HEADS, N_LAYERS, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            Self::module("layer", |s| &s.layer, |s| &mut s.layer),
            |layer| Encoder { layer },
        )
    }
}

impl<
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        const N_LAYERS: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > Module<Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>>
    for Encoder<HEAD_DIM, N_HEADS, N_LAYERS, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); HEAD_DIM * N_HEADS * 4]: Sized,
{
    type Output = Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        self.layer.try_forward(input)
    }
}

struct Pooler<const DIM: usize, E: Dtype, D: Device<E>> {
    dense: Linear<DIM, DIM, E, D>,
    activation: Tanh,
}

impl<
        const DIM: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D> for Pooler<DIM, E, D>
{
    type To<E2: Dtype, D2: Device<E2>> = Pooler<DIM, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            Self::module("dense", |s| &s.dense, |s| &mut s.dense),
            |dense| Pooler {
                dense,
                activation: Tanh,
            },
        )
    }
}

impl<
        const DIM: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > Module<Tensor<(usize, Const<DIM>), E, D>> for Pooler<DIM, E, D>
{
    type Output = Tensor<(usize, Const<DIM>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<DIM>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let applied = self.dense.try_forward(input)?;
        Ok(applied.try_slice((0..1, ..))?.realize())
    }
}

struct BertModel<
    const VOCAB_SIZE: usize,
    const MAX_TOKENS: usize,
    const HEAD_DIM: usize,
    const N_HEADS: usize,
    const N_LAYERS: usize,
    E: Dtype,
    D: Device<E>,
> where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); HEAD_DIM * N_HEADS * 4]: Sized,
{
    embeddings: BertEmbeddings<{ HEAD_DIM * N_HEADS }, VOCAB_SIZE, MAX_TOKENS, E, D>,
    encoder: Encoder<HEAD_DIM, N_HEADS, N_LAYERS, E, D>,
    pooler: Pooler<{ HEAD_DIM * N_HEADS }, E, D>,
}
impl<
        const VOCAB_SIZE: usize,
        const MAX_TOKENS: usize,
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        const N_LAYERS: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D>
    for BertModel<VOCAB_SIZE, MAX_TOKENS, HEAD_DIM, N_HEADS, N_LAYERS, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); HEAD_DIM * N_HEADS * 4]: Sized,
{
    type To<E2: Dtype, D2: Device<E2>> =
        BertModel<VOCAB_SIZE, MAX_TOKENS, HEAD_DIM, N_HEADS, N_LAYERS, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("embeddings", |s| &s.embeddings, |s| &mut s.embeddings),
                Self::module("encoder", |s| &s.encoder, |s| &mut s.encoder),
                Self::module("pooler", |s| &s.pooler, |s| &mut s.pooler),
            ),
            |(embeddings, encoder, pooler)| BertModel {
                embeddings,
                encoder,
                pooler,
            },
        )
    }
}

impl<
        const VOCAB_SIZE: usize,
        const MAX_TOKENS: usize,
        const HEAD_DIM: usize,
        const N_HEADS: usize,
        const N_LAYERS: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E> + ZerosTensor<usize>,
    > Module<Tensor<(usize,), usize, D>>
    for BertModel<VOCAB_SIZE, MAX_TOKENS, HEAD_DIM, N_HEADS, N_LAYERS, E, D>
where
    [(); HEAD_DIM * N_HEADS]: Sized,
    [(); HEAD_DIM * N_HEADS * 4]: Sized,
{
    type Output = Tensor<(usize, Const<{ HEAD_DIM * N_HEADS }>), E, D>;

    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(usize,), usize, D>) -> Result<Self::Output, Self::Error> {
        let hidden = self.embeddings.try_forward(input)?;
        let encoded = self.encoder.try_forward(hidden)?;
        Ok(encoded)
        // we are not using the pooler for MiniLM
        // self.pooler.try_forward(encoded)
    }
}

pub struct MiniLM<E: Dtype, D: Device<E>> {
    tokenizer: Tokenizer,
    model: BertModel<30522, 512, 32, 12, 6, E, D>,
}

impl<E: Dtype, D: Device<E> + ZerosTensor<usize>> Module<Tensor<(usize,), usize, D>>
    for MiniLM<E, D>
where
    E: num_traits::Float + rand_distr::uniform::SampleUniform,
    D: ZerosTensor<usize>,
{
    type Output = Tensor<Rank1<384>, E, D>;

    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(usize,), usize, D>) -> Result<Self::Output, Self::Error> {
        let token_weights = self.model.try_forward(input)?;
        let mean = token_weights.try_mean::<Rank1<384>, _>()?;
        let norm = mean.clone().try_square()?.try_sum()?.try_sqrt()?;
        mean.try_div(norm.broadcast())
    }
}

impl MiniLM<f32, Cpu> {
    // todo: error handling
    pub fn new(tokenizer_bytes: &[u8], model_bytes: &[u8]) -> Result<Self, CpuError> {
        let tokenizer =
            Tokenizer::from_bytes(tokenizer_bytes).map_err(|_| CpuError::WrongNumElements)?;

        let mut tensors =
            SafeTensors::deserialize(model_bytes).map_err(|_| CpuError::WrongNumElements)?;
        let dev = Cpu::default();

        let mut model = BertModel::<30522, 512, 32, 12, 6, f32, Cpu>::build(&dev);

        BertModel::<30522, 512, 32, 12, 6, f32, Cpu>::iter_tensors(&mut RecursiveWalker {
            m: (&mut model, String::new()),
            f: &mut tensors,
        })
        .map_err(|_| CpuError::WrongNumElements)?;
        Ok(MiniLM { tokenizer, model })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<f32>, CpuError> {
        let encoded = self
            .tokenizer
            .encode(text, true)
            .map_err(|_| CpuError::WrongNumElements)?;
        let ids: Vec<_> = encoded.get_ids().iter().map(|i| *i as usize).collect();
        let length = ids.len();
        let dev = Cpu::default();
        let ids: Tensor<(usize,), usize, Cpu> = dev.tensor_from_vec(ids, (length,));

        Ok(self.try_forward(ids)?.as_vec())
    }
}

#[cfg(test)]
mod tests {

    fn abs_diff<S: Shape, E: Dtype, D: Device<E>>(
        tensor: Tensor<S, E, D>,
        should: Tensor<S, E, D>,
    ) -> f32 {
        assert_eq!(tensor.shape().concrete(), should.shape().concrete());
        let num_elements = tensor.shape().concrete().into_iter().product::<usize>() as f32;
        tensor
            .try_sub(should)
            .expect("failed to subtract tensor from target")
            .abs()
            .as_vec()
            .iter()
            .filter_map(E::to_f32)
            .sum::<f32>()
            / num_elements
    }

    use super::*;
    use dfdx::{
        shapes::{Dtype, HasShape, Shape},
        tensor::Tensor,
        tensor_ops::{Device, TrySub},
    };

    #[cfg(feature = "pyo3")]
    #[test]
    fn test_attn_forward() {
        use dfdx::{
            prelude::{BuildModule, LoadFromSafetensors},
            shapes::{Const, HasShape},
            tensor::{Cpu, TensorFromVec, ZerosTensor},
        };
        use pyo3::Python;
        use tokenizers::Tokenizer;

        use crate::{embeddings::BertEmbeddings, tests::load_tensor};

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

        Python::with_gil(|py| {
            py.run(
                r#"
import torch as to
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
auto_model = model[0].auto_model 
layer_0 = auto_model.encoder.layer[0]
layer_0_attn = auto_model.encoder.layer[0].attention
layer_0_intermediate = auto_model.encoder.layer[0].intermediate
layer_0_output = auto_model.encoder.layer[0].output



save_file(auto_model.embeddings.state_dict(), "../python/models/embeddings.safetensors")
save_file(layer_0.state_dict(), "../python/models/layer_0.safetensors")
save_file(model[0].auto_model.state_dict(), "../python/models/model.safetensors")

tokens = to.LongTensor([model.tokenizer.encode("Hello my name is Bort")])
embedded = auto_model.embeddings(tokens)


output = layer_0(embedded)[0]
save_file({"output": output.squeeze(0)}, "../python/models/layer_0_out.safetensors")
model_out = auto_model(tokens).last_hidden_state.squeeze(0) # pooler_output
save_file({"output": model_out}, "../python/models/model_out.safetensors")

encoded = to.from_numpy(model.encode("Hello my name is Bort"))
save_file({"encoded": encoded}, "../python/models/encoded.safetensors")
                   "#,
                None,
                None,
            )
            .expect("Failed to run python");
        });

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

        let embedded = embeddings
            .try_forward(ids.clone())
            .expect("failed to call apply embeddings to ids");

        let mut layer: Layer<32, 12, f32, Cpu> = Layer::build(&dev);

        layer.load_safetensors("../python/models/layer_0.safetensors");

        let out = layer
            .try_forward(embedded)
            .expect("Failed to call self attn");

        let mut output_should: Tensor<(usize, Const<384>), f32, Cpu> = dev.zeros_like(out.shape());

        load_tensor(
            "../python/models/layer_0_out.safetensors",
            "output",
            &mut output_should,
        );

        let n_elements = out.shape().concrete().iter().product::<usize>() as f32;
        let diff = (out - output_should).abs().as_vec().iter().sum::<f32>();
        println!("DIFF: {}", diff / n_elements);
        assert!(diff / n_elements <= 1.0e-6);

        let mut model: BertModel<30522, 512, 32, 12, 6, f32, Cpu> = BertModel::build(&dev);

        model
            .load_safetensors("../python/models/model.safetensors")
            .expect("Failed to load safetensors");

        let model_out = model
            .try_forward(ids.clone())
            .expect("failed to call transformer forward");

        let mut output_should: Tensor<(usize, Const<384>), f32, Cpu> =
            dev.zeros_like(model_out.shape());

        load_tensor(
            "../python/models/model_out.safetensors",
            "output",
            &mut output_should,
        );
        let n_elements = model_out.shape().concrete().iter().product::<usize>() as f32;
        let diff = (model_out - output_should)
            .abs()
            .as_vec()
            .iter()
            .sum::<f32>();

        println!("DIFF: {}", diff / n_elements);
        assert!(diff / n_elements <= 1.0e-6);

        let minilm = MiniLM { tokenizer, model };
        let encoded = minilm.try_forward(ids).expect("Failed to encode text");

        let mut output_should: Tensor<(Const<384>,), f32, Cpu> = dev.zeros_like(encoded.shape());

        load_tensor(
            "../python/models/encoded.safetensors",
            "encoded",
            &mut output_should,
        );
        let n_elements = encoded.shape().concrete().iter().product::<usize>() as f32;
        let diff = (encoded - output_should).abs().as_vec().iter().sum::<f32>();

        println!("DIFF: {}", diff / n_elements);
    }
}

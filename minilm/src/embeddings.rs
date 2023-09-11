use dfdx::prelude::modules::{Embedding, LayerNorm1D};
use dfdx::prelude::{Dropout, Module, TensorCollection, TensorOptions};
use dfdx::shapes::{Const, Dtype, HasShape, Rank1, Rank2, Shape};
use dfdx::tensor::{Tensor, ZerosTensor};
use dfdx::tensor_ops::{Device, TryAdd};

pub struct BertEmbeddings<
    const DIM: usize,
    const VOCAB_SIZE: usize,
    const MAX_TOKENS: usize,
    E: Dtype,
    D: Device<E>,
> {
    token_embeddings: Embedding<VOCAB_SIZE, DIM, E, D>,
    position_embeddings: Embedding<MAX_TOKENS, DIM, E, D>,
    type_embeddings: Embedding<2, DIM, E, D>,
    norm: LayerNorm1D<DIM, E, D>,
    dropout: Dropout,
}

impl<
        const DIM: usize,
        const VOCAB_SIZE: usize,
        const MAX_TOKENS: usize,
        E: Dtype,
        D: Device<E>,
    > TensorCollection<E, D> for BertEmbeddings<DIM, VOCAB_SIZE, MAX_TOKENS, E, D>
{
    type To<E2: Dtype, D2: Device<E2>> = BertEmbeddings<DIM, VOCAB_SIZE, MAX_TOKENS, E2, D2>;

    fn iter_tensors<V: dfdx::prelude::ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::tensor(
                    "word_embeddings.weight",
                    |s| &s.token_embeddings.weight,
                    |s| &mut s.token_embeddings.weight,
                    TensorOptions::reset_to_ones(),
                ),
                Self::tensor(
                    "position_embeddings.weight",
                    |s| &s.position_embeddings.weight,
                    |s| &mut s.position_embeddings.weight,
                    TensorOptions::reset_to_ones(),
                ),
                Self::tensor(
                    "token_type_embeddings.weight",
                    |s| &s.type_embeddings.weight,
                    |s| &mut s.type_embeddings.weight,
                    TensorOptions::reset_to_ones(),
                ),
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
            |(wte, wpe, te, gamma, beta)| BertEmbeddings {
                token_embeddings: Embedding { weight: wte },
                position_embeddings: Embedding { weight: wpe },
                type_embeddings: Embedding { weight: te },
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
        const VOCAB_SIZE: usize,
        const MAX_TOKENS: usize,
        E: Dtype,
        D: Device<E>,
    > Module<Tensor<(usize,), usize, D>> for BertEmbeddings<DIM, VOCAB_SIZE, MAX_TOKENS, E, D>
where
    D: ZerosTensor<usize>,
{
    type Output = Tensor<(usize, Const<DIM>), E, D>;

    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(usize,), usize, D>) -> Result<Self::Output, Self::Error> {
        let dev = D::default();

        let [input_length] = input.shape().concrete();
        let positions_vec = (0..input_length).collect::<Vec<_>>();
        let positions: Tensor<(usize,), usize, D> =
            dev.tensor_from_vec(positions_vec, *input.shape());
        let types: Tensor<(usize,), usize, D> = dev.zeros_like(input.shape());

        let token_embeddings = self.token_embeddings.try_forward(input)?;
        let positions_embeddings = self.position_embeddings.try_forward(positions)?;
        let type_embeddings = self.type_embeddings.try_forward(types)?;
        let normalized = self.norm.try_forward(
            token_embeddings
                .try_add(positions_embeddings)?
                .try_add(type_embeddings)?,
        )?;
        self.dropout.try_forward(normalized)
    }
}

impl<
        const DIM: usize,
        const VOCAB_SIZE: usize,
        const MAX_TOKENS: usize,
        E: Dtype,
        D: Device<E>,
        const SEQ: usize,
    > Module<Tensor<(Const<SEQ>,), usize, D>> for BertEmbeddings<DIM, VOCAB_SIZE, MAX_TOKENS, E, D>
where
    [(); MAX_TOKENS - SEQ]: Sized,
    D: ZerosTensor<usize>,
{
    type Output = Tensor<Rank2<SEQ, DIM>, E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(Const<SEQ>,), usize, D>,
    ) -> Result<Self::Output, Self::Error> {
        let dev = D::default();

        let [input_length] = input.shape().concrete();
        let positions_vec = (0..input_length).collect::<Vec<_>>();
        let positions: Tensor<Rank1<SEQ>, usize, D> =
            dev.tensor_from_vec(positions_vec, *input.shape());
        let types: Tensor<Rank1<SEQ>, usize, D> = dev.zeros();

        let token_embeddings = self.token_embeddings.try_forward(input)?;
        let positions_embeddings = self.position_embeddings.try_forward(positions)?;
        let type_embeddings = self.type_embeddings.try_forward(types)?;
        let normalized = self
            .norm
            .try_forward(token_embeddings + positions_embeddings + type_embeddings)?;
        self.dropout.try_forward(normalized)
    }
}

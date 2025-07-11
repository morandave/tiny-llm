import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    """
    Computes the scaled dot-product attention.

    Args:
        query: Query tensor of shape (bs, H, L, D).
        key: Key tensor of shape (bs, H, L, D).
        value: Value tensor of shape (bs, H, L, D).
        scale: Scaling factor for the dot product.
        mask: Optional mask tensor to apply to the attention scores.
    Returns:
        Output tensor after applying attention.
    """
    k_T = mx.transpose(key, axes=(0, 1, 3, 2))  # (bs, H, L, D) -> (bs, H, D, L)
    scores = mx.matmul(query, k_T)  # (bs, H, L, L)
    if scale is not None:
        scores = scores * scale  # Scale the scores
    if mask is not None:
        if isinstance(mask, str):
            mask = mx.array(mask)
        scores = scores + mask  # Apply the mask
    scores = softmax(scores, axis=-1)  # Normalize the scores
    output = mx.matmul(scores, value)  # (bs, H, L, D)
    return output


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        """
        wq, wk, wv:(hidden_size, (num_heads*D) )
        wo:((num_heads*D), hidden_size)
        """
        self.pre_q_linear = linear(wq)
        self.pre_k_linear = linear(wk)
        self.pre_v_linear = linear(wv)
        self.post_linear = linear(wq)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        '''
        query:(bs,L,hidden_size)

        return;
        output:(bs, )
        '''
        scale = mx.sqrt(query.shape[-1])
        query = self.pre_q_linear(query) # (bs,L,hidden_size) -> (bs, L, H, D)
        query = mx.transpose(query, axis=(0,2,1,3)) # (bs, L, H, D) -> (bs, H, L, D)
        # ……
        attention = scaled_dot_product_attention_simple(query, key, value, scale, mask)
        attention = mx.transpose(attention, axis=(0,2,1,3))
        output = self.post_linear(attention) # ->(bs,L,hidden_size)
        return output


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass

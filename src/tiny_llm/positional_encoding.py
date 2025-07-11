import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        D_2 = self.dims / 2 # resume dims为偶数

        # 根据D_2创建旋转矩阵
        rotate = None

        # 变换x维度 (B, L, H, D)->(B, H, L, D)
        output = rotate * x

        return output

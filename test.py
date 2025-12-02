from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.mamba_encoder import MambaEncoder, EncoderArgs
import torch

input_tensor = torch.ones(size=(8, 1582, 80))
input_lens = torch.full((8,), 1582, dtype=torch.long)
encoder = ConformerEncoder(
    input_size=80, output_size=256, attention_heads=4,
    linear_units=2048, num_blocks=6, normalize_before=True,
    pos_enc_layer_type="rel_pos"
)
print(encoder.embed)
output, output_lens, _ = encoder(input_tensor, input_lens)
print(output.shape, output_lens.shape)

args = EncoderArgs(d_model=256, n_layer=2)
mamba_encoder = MambaEncoder(80, args)
print(mamba_encoder.embed)
output, output_lens, _ = mamba_encoder(input_tensor, input_lens)
print(output.shape, output_lens.shape)
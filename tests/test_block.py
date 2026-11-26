import pytest
import torch

from jabart_llm.config import ModelConfig
from jabart_llm.model.block import TransformerBlock


@pytest.fixture
def config():
    # Small config so tests run instantly; same code paths as the real 768-dim model.
    return ModelConfig(vocab_size=100, d_model=64, d_ff=256, n_heads=4, max_seq_len=32)


@pytest.fixture
def block(config):
    torch.manual_seed(0)
    return TransformerBlock(config).eval()  # eval() turns dropout off -> deterministic


def test_shape_is_preserved(block, config):
    # A block must return the same shape it receives, otherwise blocks can't be stacked.
    x = torch.randn(2, 10, config.d_model)
    out, cache = block(x)
    assert out.shape == x.shape
    assert cache is None  # use_cache=False -> no cache returned


def test_kv_cache_matches_full_pass(block, config):
    # Generating token-by-token with the cache must give the SAME result as one full pass.
    x = torch.randn(2, 8, config.d_model)
    full, _ = block(x)

    out, cache = block(x[:, :5], use_cache=True)  # prompt: first 5 tokens
    outs = [out]
    for t in range(5, 8):  # then one token at a time, like generation
        out, cache = block(x[:, t : t + 1], kv_cache=cache, use_cache=True)
        outs.append(out)

    assert torch.allclose(full, torch.cat(outs, dim=1), atol=1e-5)
    k, v = cache
    assert k.shape == (2, config.n_heads, 8, config.d_k)  # cache holds all 8 positions


def test_causal_future_does_not_leak(block, config):
    # Changing a future token must not change the outputs of earlier tokens.
    x = torch.randn(1, 6, config.d_model)
    x_changed = x.clone()
    x_changed[:, 4:] = torch.randn(1, 2, config.d_model)  # change tokens 4 and 5 only

    out, _ = block(x)
    out_changed, _ = block(x_changed)

    assert torch.allclose(out[:, :4], out_changed[:, :4], atol=1e-6)  # past: identical
    assert not torch.allclose(out[:, 4:], out_changed[:, 4:])  # future: changed


def test_residual_path_is_identity_when_sublayers_are_zero(block, config):
    # If both sub-layers output zero, the residual highway must pass x through untouched.
    with torch.no_grad():
        for linear in (block.attn.W_o, block.ffn.fc2):
            linear.weight.zero_()
            linear.bias.zero_()

    x = torch.randn(2, 5, config.d_model)
    out, _ = block(x)
    assert torch.equal(out, x)


def test_gradients_reach_every_parameter(block, config):
    # Every weight (LayerNorms, Q/K/V/O, FFN) must receive a gradient, or it can't learn.
    block.train()
    x = torch.randn(2, 5, config.d_model, requires_grad=True)
    out, _ = block(x)
    out.sum().backward()

    for name, p in block.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert p.grad.abs().sum() > 0, f"{name} gradient is all zeros"
    assert x.grad is not None


def test_dropout_only_active_in_train_mode(block, config):
    x = torch.randn(2, 5, config.d_model)

    block.eval()
    assert torch.equal(block(x)[0], block(x)[0])  # eval: same output every time

    block.train()
    assert not torch.equal(block(x)[0], block(x)[0])  # train: dropout changes it

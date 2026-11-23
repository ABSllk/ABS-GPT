from core.backend import xp
from core.nn import Module, Linear, RMSNorm, Embedding, TransformerBlock
from core.tensor import Tensor, checkpoint

class TransformerLanguageModel(Module):
    def __init__(self, vocab_size, context_size, head_num, head_size, block_num, ffn_type="relu", use_mtp=False,mtp_loss_weight=0.2, mtp_block_num=1, kv_head_num=None, use_moe=False, moe_num_experts=8, moe_top_k=2, moe_aux_weight=0.01, moe_shared_experts=False):
        super().__init__()
        self.use_checkpoint = False
        self.use_moe = use_moe
        self.moe_aux_weight = moe_aux_weight
        self.ffn_type = ffn_type
        self.use_mtp = use_mtp
        self.mtp_loss_weight = mtp_loss_weight
        self.mtp_block_num = mtp_block_num
        embedding_dim = head_num * head_size
        self.embedding = Embedding(vocab_size, embedding_dim)
        # self.position_embedding = Embedding(context_size, embedding_dim)
        self.transformer_blocks = [TransformerBlock(head_size, head_num, ffn_type=ffn_type, kv_head_num=kv_head_num, use_moe=use_moe, moe_num_experts=moe_num_experts, moe_top_k=moe_top_k, moe_shared_experts=moe_shared_experts) for _ in range(block_num)]
        self.layer_norm = RMSNorm(embedding_dim)
        self.lm_head = Linear(embedding_dim, vocab_size)
        if self.use_mtp:
            self.mtp_norm = [RMSNorm(embedding_dim) for _ in range(mtp_block_num)] # mtp层归一化
            self.mtp_block = [TransformerBlock(head_size, head_num, ffn_type=ffn_type) for _ in range(mtp_block_num)] # mtp块
            self.mtp_token_norm = [RMSNorm(embedding_dim) for _ in range(mtp_block_num)] # mtp token 归一化
            self.mtp_hidden_proj = [Linear(embedding_dim, embedding_dim) for _ in range(mtp_block_num)] # mtp隐藏层投影
            self.mtp_token_proj = [Linear(embedding_dim, embedding_dim) for _ in range(mtp_block_num)] # mtp token投影


        s = (3 / embedding_dim) ** 0.5
        self.embedding.weight.data[:] = xp.random.randn(*self.embedding.weight.data.shape) * 0.8
        self.lm_head.weight.data[:] = xp.random.randn(*self.lm_head.weight.data.shape) * 0.001

        for block in self.transformer_blocks:
            attention = block.self_attention
            for layer in (attention.W_q, attention.W_k, attention.W_v):
                layer.weight.data[:] = xp.random.uniform(-s, s, layer.weight.data.shape)
            attention.output.weight.data.fill(0)

            ffn = block.feed_forward
            if use_moe:
                experts = list(ffn.experts) 
                if ffn.shared_expert is not None:
                    experts.append(ffn.shared_expert)
            else:
                experts = [ffn]
                
            for expert in experts:
                if ffn_type == "relu" and not use_moe:
                    expert.linear1.weight.data[:] = xp.random.uniform(-0.4 * s, 0.4 * s, expert.linear1.weight.data.shape)
                    expert.linear2.weight.data.fill(0)
                else:
                    for layer in (expert.gate_linear, expert.up_linear, expert.down_linear):
                        layer.weight.data[:] = xp.random.uniform(-0.4 * s, 0.4 * s, layer.weight.data.shape)

    def __call__(self, inputs, cache=None, return_mtp=False, return_aux=False, token_mask=None, segment_ids=None, document_ids=None):
        mtp_logit = []
        x = self.embedding(inputs)
        aux_losses = []
        for i, block in enumerate(self.transformer_blocks):
            if self.use_moe and return_aux:
                if self.use_checkpoint and cache is None:
                    x, aux_loss = checkpoint(
                        block, 
                        x, 
                        return_aux=True, 
                        token_mask=token_mask, 
                        segment_ids=segment_ids
                    )
                else:
                    x, aux_loss = block(
                        x, 
                        cache=None if cache is None else cache[i], 
                        return_aux=True, 
                        token_mask=token_mask, 
                        segment_ids=segment_ids
                    )
                aux_losses.append(aux_loss)
            elif cache is None:
                x = (
                    checkpoint(
                        block,
                        x,
                        token_mask=token_mask,
                        segment_ids=segment_ids,
                    )
                    if self.use_checkpoint
                    else block(x, token_mask=token_mask, segment_ids=segment_ids)
                )
            else:
                x = block(
                    x, 
                    cache[i], 
                    token_mask=token_mask, 
                    segment_ids=segment_ids
                )
        x = self.layer_norm(x)
        main_logit = self.lm_head(x)

        if self.use_mtp and return_mtp:
            h = x
            groups = document_ids if document_ids is not None else segment_ids
            valid = (xp.asarray(token_mask, dtype=bool) if token_mask is not None
                     else xp.ones(inputs.shape, dtype=bool) if groups is not None else None)
            if groups is not None:
                groups = xp.asarray(groups)
                valid = valid & (groups >= 0)
            for i in range(self.mtp_block_num):
                depth = i + 1
                mtp_segments = None
                if valid is not None:
                    valid = valid[:, :-1]
                    if token_mask is not None:
                        valid = valid & xp.asarray(token_mask)[:, depth:]
                    if groups is not None:
                        valid = valid & (groups[:, :-depth] == groups[:, depth:])
                    positions = xp.asarray(segment_ids)[:, :-depth] if segment_ids is not None else 0
                    mtp_segments = xp.where(valid, positions, -1)
                h = h[:, :-1, :]
                token_embedding = self.embedding(inputs[:, i+1:])
                fused = (
                    self.mtp_hidden_proj[i](self.mtp_norm[i](h)) + 
                    self.mtp_token_proj[i](self.mtp_token_norm[i](token_embedding))
                )
                block = self.mtp_block[i]
                h = checkpoint(
                    block,
                    fused, 
                    token_mask=valid,
                    segment_ids=mtp_segments
                ) if self.use_checkpoint else block(fused, token_mask=valid, segment_ids=mtp_segments)
                mtp_logit.append(self.lm_head(self.layer_norm(h)))
            outputs = main_logit, mtp_logit
        else:
            outputs = main_logit

        if return_aux:
            aux_loss = sum(aux_losses) / len(aux_losses) if aux_losses else Tensor(0.0)
            return outputs, aux_loss
        return outputs

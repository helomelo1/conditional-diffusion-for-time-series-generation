import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class SimpleInfillingModel(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=512, num_layers=6, num_heads=8):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size*4,
            max_position_embeddings=512
        )
        self.encoder = BertModel(config)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        print(f"model params: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        logits = self.output_layer(hidden)

        return logits
    
    def compute_loss(self, logits, labels, mask_positions):
        batch_size, seq_len, vocab_size = logits.shape

        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        loss = F.cross_entropy(logits_flat, labels_flat, reduction='none')
        loss = loss.view(batch_size, seq_len)

        masked_loss = (loss * mask_positions.float()).sum()
        num_masked = mask_positions.sum()

        return masked_loss / num_masked if num_masked > 0 else masked_loss
    
    @torch.no_grad
    def fill_masks(self, masked_input, attention_mask):
        logits = self.forward(masked_input, attention_mask)
        filled = logits.argmax(dim=-1)

        return filled
    
    def training_step(self, batch):
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        mask_pos = batch['mask_positions']

        logits = self.forward(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels, mask_pos)

        preds = logits.argmax(dim=-1)
        correct = (preds == labels) & mask_pos
        accuracy = correct.sum().float() / mask_pos.sum().float()

        return loss, accuracy
    
# if __name__ == "__main__":
#     print("testing model\n")
    
#     model = SimpleInfillingModel(
#         vocab_size=30522,
#         hidden_size=256,
#         num_layers=4,
#         num_heads=8
#     )
    
#     # dummy batch
#     batch_size, seq_len = 4, 32
#     batch = {
#         'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
#         'labels': torch.randint(0, 1000, (batch_size, seq_len)),
#         'attention_mask': torch.ones(batch_size, seq_len),
#         'mask_positions': torch.rand(batch_size, seq_len) > 0.85
#     }
    
#     print(f"input shape: {batch['input_ids'].shape}")
#     print(f"masked tokens: {batch['mask_positions'].sum().item()}\n")
    
#     loss, accuracy = model.training_step(batch)
#     print(f"loss: {loss.item():.4f}")
#     print(f"accuracy: {accuracy.item():.4f}")
    
#     filled = model.fill_masks(batch['input_ids'], batch['attention_mask'])
#     print(f"\nfilled shape: {filled.shape}")
#     print("test passed!")
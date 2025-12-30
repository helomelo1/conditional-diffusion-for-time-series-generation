import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random
from pathlib import Path


class TextInfillingDataset(Dataset):
    def __init__(self, data_dir, max_length=128, mask_ratio=0.15, max_samples=None):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.mask_token_id = self.tokenizer.mask_token_id

        self.texts = []
        data_path = Path(data_dir)

        # loading from pos and neg folder
        for folder in ['pos', 'neg']:
            folder_path = data_path / folder
            if not folder_path.exists():
                continue

            for txt_file in folder_path.glob('*.txt'):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    text = text.replace('<br />', ' ').replace('<br/>', ' ')
                    text = ' '.join(text.split())

                    if 50 < len(text) < 5000:
                        self.texts.append(text)
                
                if max_samples and len(self.texts) >= max_samples:
                    break

            if max_samples and len(self.texts) >= max_samples:
                break

        print(f"Loaded {len(self.texts)} samples from {data_dir}")

    def __len__(self):
        return len(self.texts)
    
    def mask_tokens(self, input_ids):
        labels = input_ids.clone()
        mask_pos = torch.zeros_like(input_ids, dtype=torch.bool)

        special = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id
        ]

        for i in range(len(input_ids)):
            if input_ids[i].item() not in special:
                if random.random() < self.mask_ratio:
                    mask_pos[i] = True
                    input_ids[i] = self.mask_token_id

        return input_ids, labels, mask_pos
    
    def __getitem__(self, idx):
        text = self.texts[idx]

        enc = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        input_ids = enc['input_ids'].squeeze(0)
        attn_mask = enc['attention_mask'].squeeze(0)

        masked_ids, labels, mask_pos = self.mask_tokens(input_ids.clone())

        return {
            'input_ids': masked_ids,
            'attention_mask': attn_mask,
            'labels': labels,
            'mask_positions': mask_pos
        }
    
def get_dataloaders(data_dir="aclImdb", batch_size=16, max_length=128, max_train=None, max_test=None):
    train_data = TextInfillingDataset(
        Path(data_dir) / 'train',
        max_length=max_length,
        max_samples=max_train
    )

    test_data = TextInfillingDataset(
        Path(data_dir) / 'test',
        max_length=max_length,
        max_samples=max_test
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_data.tokenizer
    
# if __name__ == "__main__":
#     # quick test
#     train_loader, val_loader, tokenizer = get_dataloaders(
#         batch_size=4, max_train=100, max_test=20
#     )
    
#     batch = next(iter(train_loader))
#     print(f"\ntrain batches: {len(train_loader)}")
#     print(f"val batches: {len(val_loader)}")
    
#     print(f"\nbatch shapes:")
#     for k, v in batch.items():
#         print(f"  {k}: {v.shape}")
    
#     # show example
#     masked = tokenizer.decode(batch['input_ids'][0])
#     original = tokenizer.decode(batch['labels'][0], skip_special_tokens=True)
#     n_masked = batch['mask_positions'][0].sum().item()
    
#     print(f"\noriginal: {original[:150]}...")
#     print(f"masked: {masked[:150]}...")
#     print(f"masked tokens: {n_masked}")
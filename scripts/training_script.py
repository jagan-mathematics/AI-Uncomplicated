import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import logging
import torch.utils.checkpoint as checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Use gradient checkpointing here
        def compute_attention(x, mask):
            return self.self_attn(x, x, x, mask)
        
        def compute_feed_forward(x):
            return self.feed_forward(x)
        
        attn_output = checkpoint.checkpoint(compute_attention, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = checkpoint.checkpoint(compute_feed_forward, x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(DecoderOnlyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, tgt):
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        return tgt_mask & nopeak_mask

    def forward(self, tgt):
        tgt_mask = self.generate_mask(tgt)
        tgt_embedded = self.dropout(self.positional_encoding(self.embedding(tgt)))

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_mask)

        output = self.fc(dec_output)
        return output


class Trainer:
    def __init__(self, model, optimizer, criterion, dataloader, num_epochs, accumulation_steps, scheduler=None, checkpoint_path=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.accumulation_steps = accumulation_steps  # Gradient accumulation steps
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path

    def save_checkpoint(self, epoch):
        if self.checkpoint_path:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
            }
            torch.save(checkpoint, self.checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self):
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
            return checkpoint['epoch']
        return 0

    def train(self):
        start_epoch = self.load_checkpoint()  # Load checkpoint
        self.model.train()
        for epoch in range(start_epoch, self.num_epochs):
            total_loss = 0
            self.optimizer.zero_grad()  # Initialize gradients to zero at the start of each epoch
            
            for step, tgt_data in enumerate(self.dataloader):
                output = self.model(tgt_data[:, :-1])
                loss = self.criterion(output.view(-1, output.size(-1)), tgt_data[:, 1:].contiguous().view(-1))
                loss.backward()

                # Gradient accumulation
                if (step + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()  # Update weights
                    self.optimizer.zero_grad()  # Reset gradients
                total_loss += loss.item()

            if self.scheduler:
                self.scheduler.step()

            logger.info(f"Epoch: {epoch + 1}, Loss: {total_loss / len(self.dataloader)}")

            # Save checkpoint after each epoch
            self.save_checkpoint(epoch + 1)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for tgt_data in self.dataloader:
                output = self.model(tgt_data[:, :-1])  # Generate predictions
                loss = self.criterion(output.view(-1, output.size(-1)), tgt_data[:, 1:].contiguous().view(-1))
                total_loss += loss.item()

                pred = torch.argmax(output, dim=-1)
                pred_trimmed = pred[:, :tgt_data[:, 1:].size(1)]
                correct += (pred_trimmed == tgt_data[:, 1:]).sum().item()
                total += tgt_data[:, 1:].numel()

        accuracy = correct / total if total > 0 else 0
        logger.info(f"Evaluation Loss: {total_loss / len(self.dataloader)}, Accuracy: {accuracy}")


def main():
    vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 512
    dropout = 0.1
    batch_size = 32
    num_epochs = 10
    accumulation_steps = 4  # Example for gradient accumulation

    # Model, optimizer, criterion
    model = DecoderOnlyTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # DataLoader (example, should be replaced with actual data)
    dataloader = torch.utils.data.DataLoader([torch.randint(0, vocab_size, (batch_size, max_seq_length)) for _ in range(100)], batch_size=batch_size)
    
    # Trainer
    trainer = Trainer(model, optimizer, criterion, dataloader, num_epochs, accumulation_steps)
    
    # Train and evaluate
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()

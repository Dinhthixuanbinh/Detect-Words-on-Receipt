import torch

class SaveBestModelCallback:
    def __init__(self, model, save_path):
        self.model = model
        self.save_path = save_path
        self.best_val_loss = float(0.65)  

    def __call__(self, val_loss):
        if val_loss > self.best_val_loss:
            print(f"Validation loss improved ({self.best_val_loss:.6f} --> {val_loss:.6f}). Saving model...")
            torch.save(self.model.state_dict(), self.save_path)
            self.best_val_loss = val_loss
        else:
            print(f"Validation loss did not improve ({self.best_val_loss:.6f} --> {val_loss:.6f}).")
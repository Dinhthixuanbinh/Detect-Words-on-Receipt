import os
from config import get_args
from ExtractingText import ExtractingText
from Processing import CleaningData
from trainer import Trainer

def main():
    args = get_args()

    # Create instances of the classes
    extracting_text = ExtractingText(True)
    cleaning_data = CleaningData()
    trainer = Trainer(args.model_name, args.device, args.batch_size, args.num_epochs)

    # Load data
    word_position = extracting_text.TesseractOCR(sorted(os.listdir(args.data_path)),'en', de_prob=0.4)

    # Preprocess data
    train_ds, val_ds = cleaning_data.get_data(args.xml_path, word_position, split_rate=0.4)
    X_train, y_train = cleaning_data.get_NLP_data(train_ds)
    X_val, y_val = cleaning_data.get_NLP_data( val_ds)

    # Train the model
    trainer.train(X_train, y_train, X_val, y_val)

if __name__ == '__main__':
    main()

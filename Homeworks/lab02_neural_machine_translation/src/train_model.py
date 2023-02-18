import time
import tqdm
import logging
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from utils import get_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def step(model,
         optimizer,
         criterion,
         clip,
         iterator,
         trg_vocab,
         phase="train",
         device=device):

    if phase == "train":
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    start_time = time.time()
    original_text = []
    generated_text = []
    for i, batch in tqdm.tqdm(enumerate(iterator)):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output.cpu(), trg.cpu())

        if phase == "train":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        else:
            original_text.extend([get_text(x, trg_vocab) for x in trg.cpu().numpy().T])
            generated_text.extend([get_text(x, trg_vocab) for x in output[1:].detach().cpu().numpy().T])

        epoch_loss += loss.item()
    end_time = time.time()

    if phase == "train":
        return epoch_loss / len(iterator), end_time - start_time

    bleu_score = corpus_bleu([[text] for text in original_text], generated_text) * 100
    return epoch_loss / len(iterator), bleu_score, end_time - start_time


def train_model(model,
                optimizer,
                criterion,
                train_iterator,
                valid_iterator,
                logging_file,
                best_model_path,
                clip,
                num_epochs
                ):

    logging.basicConfig(filename=logging_file,
                        filemode='a',
                        format='%(message)s',
                        level=logging.INFO)
    logger = logging.getLogger()

    best_bleu = -1
    start_train = time.time()

    logger.info("Start training model...")
    for num_epoch in range(num_epochs):
        logging.info(f"epoch: {num_epoch + 1}")
        train_loss, train_time = step(model, optimizer, criterion, clip, train_iterator, phase="train")
        valid_loss, bleu, valid_time = step(model, optimizer, criterion, clip, valid_iterator, phase="valid")

        if bleu > best_bleu:
            best_bleu = bleu
            best_ppl = np.exp(valid_loss)
            torch.save(model.state_dict(), best_model_path)

        logger.info(
            f"train time:  {(train_time // 60):.0f} m {(train_time % 60):.0f} s, loss: {train_loss:.3f}, PPL: {np.exp(train_loss):.3f}")
        logger.info(
            f"valid time: {(valid_time // 60):.0f} m {(valid_time % 60):.0f} s, loss: {valid_loss:.3f}, PPL: {np.exp(valid_loss):.3f}, bleu: {bleu:.3f} \n")
    end_train = time.time()
    time_elapsed = start_train - end_train
    logger.info(f"Training complete in {(time_elapsed // 60):.0f} m {(time_elapsed % 60):.0f} s")
    logger.info(f"Model state saved at {best_model_path}, the best bleu score is: {best_bleu:.3f}, PPL: {best_ppl:.3f}")
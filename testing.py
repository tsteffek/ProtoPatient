import fire
import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import RandomSampler
from transformers import AutoTokenizer

from dataset.outcome import OutcomeDiagnosesDataset, collate_batch
from model.proto import ProtoModule


def run_tests(test_file,
              batch_size=10,
              gpus=1,
              max_length=512,
              check_val_every_n_epoch=1,
              num_val_samples=None,
              pretrained_model='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
              save_dir='output',
              resume_from_checkpoint=None,
              seed=7,
              all_labels_path=None,
              few_shot_experiment=False,
              label_column='short_codes',
              id_column='id',
              **ignored_kwargs):
    pl.seed_everything(seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if few_shot_experiment:
        dataset = FilteredDiagnosesDataset
    else:
        dataset = OutcomeDiagnosesDataset

    test_dataset = dataset(test_file, tokenizer, max_length=max_length, all_codes_path=all_labels_path, label_column=label_column, id_column=id_column)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                             collate_fn=collate_batch,
                                             batch_size=batch_size,
                                             num_workers=10,
                                             pin_memory=True,
                                             shuffle=False,
                                             sampler=RandomSampler(test_dataset,
                                                                   replacement=True,
                                                                   num_samples=num_val_samples))

    tb_logger = TensorBoardLogger(save_dir, name="lightning_logs")

    trainer = pl.Trainer(logger=tb_logger,
                         default_root_dir=save_dir,
                         gpus=gpus,
                         check_val_every_n_epoch=check_val_every_n_epoch,
                         deterministic=True,
                         accelerator="ddp",
                         resume_from_checkpoint=resume_from_checkpoint
                         )

    trainer.test(dataloaders=test_dataloader, model=ProtoModule.load_from_checkpoint(resume_from_checkpoint))


if __name__ == '__main__':
    fire.Fire(run_tests)

device = 'cuda'

num_ep = 100
optimizer_config = dict(
    type='adamw',
    config = dict(
        lr = 3e-3,
        weight_decay=1.0
        # momentum=0.9,
    )
)

lr_sche_config = dict(
    type = 'constant',
    config = dict(
        # epoches=[60, 80],
        # muls=[0.1, 0.1]
    )
)

model_type = 'modern_model'
model_config = dict(
    max_len = 512,
    head_dim=32,
    num_head=8,
    dim_ffn=1024,
    num_kv_groups=4,
    dropout=0.1,
    num_layers=12,
    num_classes=5004
)



cifar_data_root = 'DATA'
train_data_config = dict(
    pad_token_idx=0,
    max_len = 512,
    dataset_config = dict(
        data_pkl_p='DATA/pretoknized_train_data.pkl',
        word2idx_pkl_p='DATA/word2idx.pkl', 
        idx2word_pkl_p='DATA/idx2word.pkl',
    ), 
    data_loader_config = dict(
        batch_size = 1024,
        num_workers = 4,
        shuffle=True
    )
)
test_data_config = dict(
    pad_token_idx=0,
    max_len = 512,
    dataset_config = dict(
        data_pkl_p='DATA/pretoknized_val_data.pkl',
        word2idx_pkl_p='DATA/word2idx.pkl', 
        idx2word_pkl_p='DATA/idx2word.pkl',
    ), 
    data_loader_config = dict(
        batch_size = 16,
        num_workers = 4,
        shuffle=False
        
    )
)



resume_ckpt_path = None
load_weight_from = None

# ckp
ckp_config = dict(
   save_last=None, 
   every_n_epochs=None,
#    monitor='val_mae',
#    mode='min',
#    filename='{epoch}-{val_mae:.3f}'
)

# trainer config
trainer_config = dict(
    log_every_n_steps=5,
    precision='32',
    # val_check_interval=0.4, # val after k training batch 0.0-1.0, or a int
    check_val_every_n_epoch=1,
    gradient_clip_val=1.0
)


# LOGGING
enable_wandb = True
wandb_config = dict(
    project = 'backbone-exp',
    offline = True
)
ckp_root = f'[{wandb_config["project"]}]'
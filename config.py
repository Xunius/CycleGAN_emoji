config = {
        'train_path': './data/',
        'test_path': './data/',
        'output_dir': './outputs/exp1',

        'image_size': 64,
        'img_channels': 3,
        'n_features': 32,
        'n_resblocks': 5,
        'n_strides': 2,

        'batch_size': 32,
        'lambda_dis': 1,
        'lambda_gen': 1,
        'lambda_cycle': 10,
        'lambda_identity': 3,
        'p_cycle': 0.9,
        'p_identity': 0.5,
        'progressive': False,   # progressively train each of decoder conv layer or not
        'progressive_n_stage': 10,  # if <progressive>, number of iterations each decoder layer is trained before releasing the training of the next layer
        'n_dis_per_gen': 2,     # number of discriminator training per generator training iteration

        'n_iters_eval': 500,
        'n_eval_samples': 4,

        'resume': False,
        'resume_from_epoch': 99,
        'device': None,
        'n_epoch_save': 20,
        'n_epochs': 100,

        'lr0': 1e-4,
        'warmup_iter': 1000,
        'peak_lr': 1.0e-3,
        'weight_decay': 1e-4,

        'dis_beta1': 0.5,
        'gen_beta1': 0.5,
        'dropout': 0.2,
        }

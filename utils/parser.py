from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser()
    
    parser.add_argument("--target_language")
    parser.add_argument("--num_neg", type=int, default=2)
    parser.add_argument("--train_datapath", default="data/aligned/en2zh/en.zh.train.jl")
    parser.add_argument("--val_datapath", default="data/aligned/en2zh/en.zh.dev.jl")
    parser.add_argument("--test_datapath", default="data/aligned/en2zh/en.zh.test1.jl")
    parser.add_argument("--train_target_datapath", default="data/aligned/en2zh/en.zh.train.jl")
    parser.add_argument("--val_target_datapath", default="data/aligned/en2zh/en.zh.dev.jl")
    parser.add_argument("--test_target_datapath", default="data/aligned/en2zh/en.zh.test1.jl")
    
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--resume_path", default=None)
    parser.add_argument("--output_path", default=None)
    
    parser.add_argument("--model_name", default="bert-base-multilingual-cased")

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dense_dim", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--num_warmup_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--lr_step_size", type=int, default=5)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=int, default=0.0005)

    parser.add_argument("--use_gpu", action="store_true", default=True)
    # parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    
    return parser
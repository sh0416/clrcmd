from clrcmd.models import create_contrastive_learning
from clrcmd.trainer import STSTrainer

model = create_contrastive_learning("bert-rcmd", 0.05)
model.train()
print(model)
model.model.representation_model.push_to_hub("sh0416/clrcmd")
exit()

trainer = STSTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbackes=[],
)

import os
import torch
from torch.utils.data import Dataset


class DreamBoothDataset(Dataset):

    def __init__(
        self,  folder,  have_pooled_prompt_embeds = True
    ):

        self.positive_prompt_embeds = []
        self.have_pooled_prompt_embeds = have_pooled_prompt_embeds
        if self.have_pooled_prompt_embeds:
            self.positive_pooled_prompt_embeds = []
        self.latents = []
      
    
        prompt_seeds = next(os.walk(folder))[1] 


        for prompt_seed in prompt_seeds:
            
            folder_save = f"{folder}/{prompt_seed}/"
        
            positive_prompt_embeds = folder_save+"/prompt_embeds.pt"
            if self.have_pooled_prompt_embeds:
                positive_pooled_prompt_embeds = folder_save+"/pooled_prompt_embeds.pt"
            latents = folder_save+"/latents.pt"

            self.positive_prompt_embeds.append(positive_prompt_embeds)
            if self.have_pooled_prompt_embeds:
                self.positive_pooled_prompt_embeds.append(positive_pooled_prompt_embeds)
            self.latents.append(latents)

        self._length = len(self.latents)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        example["positive_prompt_embeds"] = torch.load(
            self.positive_prompt_embeds[index], 
            weights_only=True)
        if self.have_pooled_prompt_embeds:
            example["positive_pooled_prompt_embeds"] = torch.load(
                self.positive_pooled_prompt_embeds[index], 
                weights_only=True)
        example["latents"] = torch.load(
            self.latents[index], 
            weights_only=True)
        
        return example


def collate_fn(examples):
    positive_prompt_embeds = [example["positive_prompt_embeds"] for example in examples]
    if "positive_pooled_prompt_embeds" in examples[0].keys():
        positive_pooled_prompt_embeds = [example["positive_pooled_prompt_embeds"] for example in examples]
    latents = [example["latents"] for example in examples]

    # torch.stack
    positive_prompt_embeds = torch.cat(positive_prompt_embeds).to(memory_format=torch.contiguous_format).float()
    if "positive_pooled_prompt_embeds" in examples[0].keys():
        positive_pooled_prompt_embeds = torch.cat(positive_pooled_prompt_embeds).to(memory_format=torch.contiguous_format).float()
    latents = torch.cat(latents).to(memory_format=torch.contiguous_format).float()

    batch = {
             "prompt_embeds": positive_prompt_embeds, 
             "latents": latents,
            }
    if "positive_pooled_prompt_embeds" in examples[0].keys():
        batch["pooled_prompt_embeds"] = positive_pooled_prompt_embeds

    return batch

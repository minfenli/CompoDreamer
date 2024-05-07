import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
import open_clip
import torchvision.transforms as T

class SDS:
    """
    Class to implement the SDS loss function.
    """

    def __init__(
        self,
        sd_version="2.1",
        device="cpu",
        t_range=[0.02, 0.98],
        output_dir="output",
    ):
        """
        Load the Stable Diffusion model and set the parameters.

        Args:
            sd_version (str): version for stable diffusion model
            device (_type_): _description_
        """

        # Set the stable diffusion model key based on the version
        if sd_version == "2.1":
            sd_model_key = "stabilityai/stable-diffusion-2-1-base"
        else:
            raise NotImplementedError(
                f"Stable diffusion version {sd_version} not supported"
            )

        # Set parameters
        self.H = 512  # default height of Stable Diffusion
        self.W = 512  # default width of Stable Diffusion
        self.num_inference_steps = 50
        self.output_dir = output_dir
        self.device = device
        self.precision_t = torch.float32

        # Create model
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_key, torch_dtype=self.precision_t
        ).to(device)

        self.preprocess = T.Resize((self.H, self.W))
        self.vae = sd_pipe.vae
        self.tokenizer = sd_pipe.tokenizer
        self.text_encoder = sd_pipe.text_encoder
        self.unet = sd_pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            sd_model_key, subfolder="scheduler", torch_dtype=self.precision_t
        )
        del sd_pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device
        )  # for convenient access

        print(f"[INFO] loaded stable diffusion!")

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        """
        Get the text embeddings for the prompt.

        Args:
            prompt (list of string): text prompt to encode.
        """
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    def encode_imgs(self, img):
        """
        Encode the image to latent representation.

        Args:
            img (tensor): image to encode. shape (N, 3, H, W), range [0, 1]

        Returns:
            latents (tensor): latent representation. shape (1, 4, 64, 64)
        """
        # check the shape of the image should be 512x512
        assert img.shape[-2:] == (512, 512), "Image shape should be 512x512"

        img = 2 * img - 1  # [0, 1] => [-1, 1]

        img = self.preprocess(img)
        posterior = self.vae.encode(img).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def decode_latents(self, latents):
        """
        Decode the latent representation into RGB image.

        Args:
            latents (tensor): latent representation. shape (1, 4, 64, 64), range [-1, 1]

        Returns:
            imgs[0] (np.array): decoded image. shape (512, 512, 3), range [0, 255]
        """
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents.type(self.precision_t)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)  # [-1, 1] => [0, 1]
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()  # torch to numpy
        imgs = (imgs * 255).round()  # [0, 1] => [0, 255]
        return imgs[0]

    def sds_loss(
        self,
        latents,
        text_embeddings,
        text_embeddings_uncond=None,
        guidance_scale=100,
        grad_scale=1
    ):
        """
        Compute the SDS loss.

        Args:
            latents (tensor): input latents, shape [1, 4, 64, 64]
            text_embeddings (tensor): conditional text embedding (for positive prompt), shape [1, 77, 1024]
            text_embeddings_uncond (tensor, optional): unconditional text embedding (for negative prompt), shape [1, 77, 1024]. Defaults to None.
            guidance_scale (int, optional): weight scaling for guidance. Defaults to 100.
            grad_scale (int, optional): gradient scaling. Defaults to 1.

        Returns:
            loss (tensor): SDS loss
        """

        # sample a timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            (latents.shape[0],),
            dtype=torch.long,
            device=self.device,
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            ### YOUR CODE HERE ###
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            noise_pred = self.unet(latents_noisy, t, text_embeddings).sample
 
            if text_embeddings_uncond is not None and guidance_scale != 1:
                ### YOUR CODE HERE ###
                noise_pred_uncond = self.unet(latents_noisy, t, text_embeddings_uncond).sample
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            w = 1 - self.alphas[t]
            gradient = w * (noise_pred - noise)
            latents_target = latents - grad_scale * gradient

        loss = ((latents_target - latents) ** 2).sum()

        return loss

    def sds_loss_batch(
        self,
        latents,
        text_embeddings,
        text_embeddings_uncond=None,
        guidance_scale=100,
        grad_scale=1
    ):
        loss = 0.
        for i in range(len(latents)):
            loss += self.sds_loss(latents[i:i+1, ...], text_embeddings, text_embeddings_uncond, guidance_scale, grad_scale)
        if len(latents) > 0:
            loss /= len(latents)
        return loss


class CLIP:
    """
    Class to implement the SDS loss function.
    """

    def __init__(
        self,
        device="cpu",
        output_dir="output",
    ):
        """
        Load the Stable Diffusion model and set the parameters.

        Args:
            sd_version (str): version for stable diffusion model
            device (_type_): _description_
        """

        # Set parameters
        self.H = 224  # default height of CLIP
        self.W = 224  # default width of CLIP
        self.output_dir = output_dir
        self.device = device

        # Set the open_clip model key based on the version
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.preprocess = T.Compose([T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                     T.Resize((self.H, self.W))])
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model = model.to(device)

        print(f"[INFO] loaded OpenClip!")

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        """
        Get the text embeddings for the prompt.

        Args:
            prompt (list of string): text prompt to encode.
        """
        return self.model.encode_text(self.tokenizer(prompt).to(self.device))

    def encode_imgs(self, image):
        """
        Encode images to latent representation.

        Args:
            img (tensor): image to encode. shape (N, 3, H, W), range [0, 1]

        Returns:
            latents (tensor): latent representation. shape (N, 512)
        """

        image = self.preprocess(image)

        # Encode the rendered image to latents
        image_embeddings = self.model.encode_image(image)

        return image_embeddings 

    def clip_loss(
        self,
        imgs,
        text_embeddings,
        text_embeddings_uncond=None
    ):
        """
        Compute the SDS loss.

        Args:
            imgs (tensor): input latents, shape [N, H, W, 3]
            text_embeddings (tensor): conditional text embedding (for positive prompt), shape [1, 77, 1024]
            text_embeddings_uncond (tensor, optional): unconditional text embedding (for negative prompt), shape [1, 77, 1024]. Defaults to None.

        Returns:
            loss (tensor): CLIP loss
        """
        image_embeddings = self.encode_imgs(imgs)
        # Compute the loss
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        if text_embeddings_uncond is not None:
            text_embeddings_uncond = text_embeddings_uncond / text_embeddings_uncond.norm(dim=-1, keepdim=True)
            text_embeddings = torch.cat([text_embeddings, text_embeddings_uncond])
            text_probs = (image_embeddings @ text_embeddings.T).mean(0)
            loss = -text_probs[0] + text_probs[1:].mean()
        else:
            text_probs = (image_embeddings @ text_embeddings.T).mean(0)
            loss = -text_probs[0]
            
        return loss

    def clip_score(
        self,
        imgs,
        text_embeddings
    ):
        """
        Compute the SDS loss.

        Args:
            imgs (tensor): input latents, shape [N, H, W, 3]
            text_embeddings (tensor): conditional text embedding (for positive prompt), shape [1, 77, 1024]
            text_embeddings_uncond (tensor, optional): unconditional text embedding (for negative prompt), shape [1, 77, 1024]. Defaults to None.

        Returns:
            loss (tensor): CLIP loss
        """
        image_embeddings = self.encode_imgs(imgs)
        # Compute the loss
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        text_probs = (image_embeddings @ text_embeddings.T).mean(0)
            
        return text_probs
import torch
import numpy as np

def robust_hidden(cover_image, bitstream):
    print(cover_image.shape) # [1, 3, 512, 512]
    print(bitstream.shape) # [1, 64]
    from models.robust.hidden.hidden_utils import model_from_checkpoint
    from models.robust.hidden.hidden_model.hidden import Hidden
    from models.robust.hidden.hidden_noise_layers.noiser import Noiser
    from models.robust.hidden.hidden_options import HiDDenConfiguration

    h, w = cover_image.shape[-2:]
    # message_length = bitstream.shape[1]  # should use this
    # TODO: bistream length not match
    message_length = 30
    CHECKPOINT_FILE = '/home/lai/Research/Graduate/HiDDeN/runs/no-noise 2024.12.11--17-34-47/checkpoints/no-noise--epoch-99.pyt'
    device = torch.device('cuda:0')

    # Construct model
    hidden_config = HiDDenConfiguration(H=h, W=w,
                                    message_length=message_length,
                                    encoder_blocks=4, encoder_channels=64,
                                    decoder_blocks=7, decoder_channels=64,
                                    use_discriminator=True,
                                    use_vgg=False,
                                    discriminator_blocks=3, discriminator_channels=64,
                                    decoder_loss=1,
                                    encoder_loss=0.7,
                                    adversarial_loss=1e-3,
                                    enable_fp16=False
                                    )
    noise_config = []
    noiser = Noiser(noise_config, device=device)

    checkpoint = torch.load(CHECKPOINT_FILE)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    model_from_checkpoint(hidden_net, checkpoint)
    # print(hidden_net.to_stirng())

    # Pre-processing: Range conversion
    # Transform from [0, 1] to [-1, 1]
    cover_image = cover_image * 2 - 1
    # Transform from [-0.5, 0.5] to [0, 1]
    message = bitstream[:, :message_length]
    message = message + 0.5

    # losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([cover_image, message])
    hidden_net.encoder.eval()
    hidden_net.noiser.eval()
    hidden_net.decoder.eval()
    hidden_net.discriminator.eval()
    with torch.no_grad():
        encoded_images = hidden_net.encoder(cover_image, message)
        ######### ROI start #########
        remove_ratio = 0.7 # this is ROI

        center_size = int(512 * remove_ratio)
        start_x = (512 - center_size) // 2
        start_y = start_x

        clear_center = cover_image[:, :, start_x:start_x + center_size, start_y:start_y + center_size]

        encoded_images[:, :, start_x:start_x + center_size, start_y:start_y + center_size] = clear_center
        ######### ROI end #########
        noised_and_cover = hidden_net.noiser([encoded_images, cover_image])
        noised_images = noised_and_cover[0]
        decoded_messages = hidden_net.decoder(noised_images)

    # Post-processing
    decoded_rounded = decoded_messages.detach().round().clip(0, 1)
    message_detached = message.detach()
    print('original: {}'.format(message_detached))
    print('decoded : {}'.format(decoded_rounded))
    print('error : {:.3f}'.format(torch.mean(torch.abs(decoded_rounded - message_detached))))
    

    encoded_images = encoded_images * 0.5 + 0.5
    decoded_rounded = decoded_rounded - 0.5
    return encoded_images, decoded_rounded
    


def decode_hidden(container_image):
    print(container_image.shape) # [1, 3, 512, 512]
    # print(container_image.min(), container_image.max()) # [0, 1]
    from models.robust.hidden.hidden_utils import model_from_checkpoint
    from models.robust.hidden.hidden_model.hidden import Hidden
    from models.robust.hidden.hidden_noise_layers.noiser import Noiser
    from models.robust.hidden.hidden_options import HiDDenConfiguration

    h, w = container_image.shape[-2:]
    # message_length = bitstream.shape[1]  # should use this
    message_length = 30
    CHECKPOINT_FILE = '/home/lai/Research/Graduate/HiDDeN/runs/no-noise 2024.12.11--17-34-47/checkpoints/no-noise--epoch-99.pyt'
    device = torch.device('cuda:0')

    # Construct model
    hidden_config = HiDDenConfiguration(H=h, W=w,
                                    message_length=message_length,
                                    encoder_blocks=4, encoder_channels=64,
                                    decoder_blocks=7, decoder_channels=64,
                                    use_discriminator=True,
                                    use_vgg=False,
                                    discriminator_blocks=3, discriminator_channels=64,
                                    decoder_loss=1,
                                    encoder_loss=0.7,
                                    adversarial_loss=1e-3,
                                    enable_fp16=False
                                    )
    noise_config = []
    noiser = Noiser(noise_config, device=device)

    checkpoint = torch.load(CHECKPOINT_FILE)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    model_from_checkpoint(hidden_net, checkpoint)
    # print(hidden_net.to_stirng())

    # Pre-processing: Range conversion
    # Transform from [0, 1] to [-1, 1]
    container_image = container_image * 2 - 1
    # Transform from [-0.5, 0.5] to [0, 1]
    message = torch.randint(0, 2, (message_length,), device='cuda') # TODO

    # losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([container_image, message])
    hidden_net.noiser.eval()
    hidden_net.decoder.eval()
    hidden_net.discriminator.eval()
    with torch.no_grad():
        noised_and_cover = hidden_net.noiser([container_image, container_image])
        noised_images = noised_and_cover[0]
        decoded_messages = hidden_net.decoder(noised_images)

    # Post-processing
    decoded_rounded = decoded_messages.detach().round().clip(0, 1)
    message_detached = message.detach()
    print('original: {}'.format(message_detached))
    print('decoded : {}'.format(decoded_rounded))
    print('error : {:.3f}'.format(torch.mean(torch.abs(decoded_rounded - message_detached))))
    
    return
    


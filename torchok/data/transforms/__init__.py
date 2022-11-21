from albumentations import (Normalize, Resize, HorizontalFlip, VerticalFlip, Affine, CenterCrop, CoarseDropout, Crop,
                            CropAndPad, CropNonEmptyMaskIfExists, ElasticTransform, Flip, GridDistortion, GridDropout,
                            Lambda, LongestMaxSize, MaskDropout, NoOp, OpticalDistortion, PadIfNeeded, Perspective,
                            PiecewiseAffine, PixelDropout, RandomCrop, RandomCropNearBBox, RandomGridShuffle,
                            RandomResizedCrop, RandomRotate90, RandomScale, RandomSizedBBoxSafeCrop, RandomSizedCrop,
                            Rotate, SafeRotate, ShiftScaleRotate, SmallestMaxSize, Transpose, AdvancedBlur, Blur,
                            CLAHE, ChannelDropout, ChannelShuffle, ColorJitter, Downscale, Emboss, Equalize, FDA,
                            FancyPCA, FromFloat, GaussNoise, GaussianBlur, GlassBlur, HistogramMatching, ISONoise,
                            HueSaturationValue, ImageCompression, InvertImg, MedianBlur, MotionBlur, Posterize,
                            MultiplicativeNoise, PixelDistributionAdaptation, RGBShift, RandomBrightnessContrast,
                            RandomFog, RandomGamma, RandomRain, RandomShadow, RandomSunFlare, RandomSnow, Sharpen,
                            RandomToneCurve, RingingOvershoot, TemplateTransform, Superpixels, Solarize, ToFloat,
                            ToGray, ToSepia, UnsharpMask)

from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.composition import Compose, OneOf

from torchok.constructor import TRANSFORMS
from torchok.data.transforms import spatial
from torchok.data.transforms import pixelwise


TRANSFORMS.register_class(Compose)
TRANSFORMS.register_class(OneOf)
TRANSFORMS.register_class(ToTensorV2)
TRANSFORMS.register_class(Normalize)
TRANSFORMS.register_class(Resize)

TRANSFORMS.register_class(HorizontalFlip)
TRANSFORMS.register_class(VerticalFlip)
TRANSFORMS.register_class(Affine)
TRANSFORMS.register_class(CenterCrop)
TRANSFORMS.register_class(CoarseDropout)
TRANSFORMS.register_class(Crop)
TRANSFORMS.register_class(CropAndPad)
TRANSFORMS.register_class(CropNonEmptyMaskIfExists)
TRANSFORMS.register_class(ElasticTransform)
TRANSFORMS.register_class(Flip)
TRANSFORMS.register_class(GridDistortion)
TRANSFORMS.register_class(GridDropout)
TRANSFORMS.register_class(Lambda)
TRANSFORMS.register_class(LongestMaxSize)
TRANSFORMS.register_class(MaskDropout)
TRANSFORMS.register_class(NoOp)
TRANSFORMS.register_class(OpticalDistortion)
TRANSFORMS.register_class(PadIfNeeded)
TRANSFORMS.register_class(Perspective)
TRANSFORMS.register_class(PiecewiseAffine)
TRANSFORMS.register_class(PixelDropout)
TRANSFORMS.register_class(RandomCrop)
TRANSFORMS.register_class(RandomCropNearBBox)
TRANSFORMS.register_class(RandomGridShuffle)
TRANSFORMS.register_class(RandomResizedCrop)
TRANSFORMS.register_class(RandomRotate90)
TRANSFORMS.register_class(RandomScale)
TRANSFORMS.register_class(RandomSizedBBoxSafeCrop)
TRANSFORMS.register_class(RandomSizedCrop)
TRANSFORMS.register_class(Rotate)
TRANSFORMS.register_class(SafeRotate)
TRANSFORMS.register_class(ShiftScaleRotate)
TRANSFORMS.register_class(SmallestMaxSize)
TRANSFORMS.register_class(Transpose)

TRANSFORMS.register_class(AdvancedBlur)
TRANSFORMS.register_class(Blur)
TRANSFORMS.register_class(CLAHE)
TRANSFORMS.register_class(ChannelDropout)
TRANSFORMS.register_class(ChannelShuffle)
TRANSFORMS.register_class(ColorJitter)
TRANSFORMS.register_class(Downscale)
TRANSFORMS.register_class(Emboss)
TRANSFORMS.register_class(Equalize)
TRANSFORMS.register_class(FDA)
TRANSFORMS.register_class(FancyPCA)
TRANSFORMS.register_class(FromFloat)
TRANSFORMS.register_class(GaussNoise)
TRANSFORMS.register_class(GaussianBlur)
TRANSFORMS.register_class(GlassBlur)
TRANSFORMS.register_class(HistogramMatching)
TRANSFORMS.register_class(HueSaturationValue)
TRANSFORMS.register_class(ISONoise)
TRANSFORMS.register_class(ImageCompression)
TRANSFORMS.register_class(InvertImg)
TRANSFORMS.register_class(MedianBlur)
TRANSFORMS.register_class(MotionBlur)
TRANSFORMS.register_class(MultiplicativeNoise)
TRANSFORMS.register_class(PixelDistributionAdaptation)
TRANSFORMS.register_class(Posterize)
TRANSFORMS.register_class(RGBShift)
TRANSFORMS.register_class(RandomBrightnessContrast)
TRANSFORMS.register_class(RandomFog)
TRANSFORMS.register_class(RandomGamma)
TRANSFORMS.register_class(RandomRain)
TRANSFORMS.register_class(RandomShadow)
TRANSFORMS.register_class(RandomSnow)
TRANSFORMS.register_class(RandomSunFlare)
TRANSFORMS.register_class(RandomToneCurve)
TRANSFORMS.register_class(RingingOvershoot)
TRANSFORMS.register_class(Sharpen)
TRANSFORMS.register_class(Solarize)
TRANSFORMS.register_class(Superpixels)
TRANSFORMS.register_class(TemplateTransform)
TRANSFORMS.register_class(ToFloat)
TRANSFORMS.register_class(ToGray)
TRANSFORMS.register_class(ToSepia)
TRANSFORMS.register_class(UnsharpMask)

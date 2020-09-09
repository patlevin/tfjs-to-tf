# Test dataset for image classification

## Content

The dataset consists of about 1300 CGI images of humans and horses.
The original images have been resized to 32x32 pixels and saved in
JPEG format using medium compression using `imagemagick`:

```sh
morgify -resize 32x32 -format jpg -quality 82 ./train/horses/*.png
morgify -resize 32x32 -format jpg -quality 82 ./train/humans/*.png
morgify -resize 32x32 -format jpg -quality 82 ./test/horses/*.png
morgify -resize 32x32 -format jpg -quality 82 ./test/humans/*.png
```

The dataset keeps the original folder structure and -split between
training and validation sets. Labels can be inferred from folder names.

## Credits

### Dataset

|     |      |
| --- | :--- |
| **Author** | Laurence Moroney                                    |
| **Title**  | "Horses or Humans Dataset"                          |
| **Date**   | February 2019                                       |
| **URL**    | [laurencemoroney.com/horses-or-humans-dataset](http://laurencemoroney.com/horses-or-humans-dataset) |

### Test images

| File | Source |
| :--  | :--    |
| horse.jpg     | [en.wikipedia.org/wiki/Zaniskari](https://en.wikipedia.org/wiki/Zaniskari#/media/File:Zaniskari_Horse_in_Ladhak,_Jammu_and_kashmir.jpg) |
| horse1.jpg    | [en.wikipedia.org/wiki/Friesian_horse](https://en.wikipedia.org/wiki/Friesian_horse#/media/File:Friesian_Stallion.jpg) |
| human.jpg     | [en.wikipedia.org/wiki/Osteoporosis](https://en.wikipedia.org/wiki/Osteoporosis#/media/File:OsteoCutout.png) |
| human1.jpg    | [en.wikipedia.org/wiki/Running](https://en.wikipedia.org/wiki/Running#/media/File:How_to_achieve_your_weight_loss_goals.jpg) |

**all test images were cropped and resized*

# trickster-GAN

This is the quarter project for ECE-285 by Sri Shatagopam and Michael Baluja.

Facial recognition software is utilized in many different applications, by both commercial and government entities. Many of these applications have the possibility to pose a risk to an individuals' freedom.  There already exist techniques to fool these kinds of software to prevent the hampering of these freedoms, from static white-box attack algorithms to entire deep neural networks that add generated noise to an image to trick existing facial recognition systems. We introduce and implement trickster-GAN, a network that utilizes adversarial attack mechanics to fool facial recognition software; this model is a modified AI-GAN run on the UTKFace facial recognition dataset. When attacking a ResNet-18 model trained on the UTKFace dataset, the classifier test accuracy drops from over 98\% to as low as 33\%.

Demo available here: 

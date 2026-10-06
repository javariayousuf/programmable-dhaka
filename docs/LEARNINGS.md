# Tips and learnings

What I'd tell someone trying this.

- **Fine-tuning pays off fast when the new footage looks like what you trained on.** In Test 1 (see the README), 141 labeled pictures and about 10 to 15 minutes of training on a laptop took rickshaws from 45 found out of 163 to about 130.
- **The model seems strongest on scenes like the ones it was shown.** In Test 2, a very different scene was harder, and the original pipeline scored higher there. Collecting the odd cases on purpose might help: other kinds of carts, night, rain, partly hidden vehicles.
- **Give look-alikes their own name.** Without a "car" and a "truck", they get called the thing you care about.
- **Label every object, and check for the ones the model missed.** A review page that only shows what the model found can never add a missing box. On one test clip it hid half of the rickshaws, and a crowded clip had so many missing that I removed it. A page for drawing missing boxes fixed it.
- **A few clear, complete clips beat many partial ones.** Adding clips with few rickshaws did not move rickshaw scores. One steady clip with every rickshaw labeled did (see the experiment log).
- **Test on footage you set aside,** and say what confidence cutoff you used.
- **The speed is a good start.** About 30 milliseconds per picture on a laptop. I have not tested an edge device, and that is the next thing I would check.

Roboflow's own guide, [How to Improve Your Computer Vision Model](https://roboflow.com/blog/how-to-improve-your-computer-vision-model), covers starting with 50 to 100 images, collecting data across times of day and weather, and tuning the confidence threshold.
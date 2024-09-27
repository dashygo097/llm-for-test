import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset,load_dataset
from peft import LoraConfig, get_peft_model


model_name = "./gemma-2b-it-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# data with prefix
'''
# zero-shot
'''
data = {
    "text": [
        "Classify the following sentence as positive or negative:\n\nSentence: This sentence is Positive.\nAnswer: ",
        "Classify the following sentence as positive or negative:\n\nSentence: This sentence is Negative.\nAnswer: ",

    ],
    "label": [
        "Positive\n",
        "Negative\n",
    ]
}
'''

'''
# 2-shot 
data = {
    "text": [
        "Classify the following sentence as positive or negative:\n\nSentence: 'I love this movie.'\nAnswer: ",
        "Classify the following sentence as positive or negative:\n\nSentence: 'I do like the main characters.'\nAnswer: ",
        "Classify the following sentence as positive or negative:\n\nSentence: 'I hate this movie.'\nAnswer: ",
        "Classify the following sentence as positive or negative:\n\nSentence: 'How can a movie be awful like this?'\nAnswer: ",

    ],
    "label": [
        "Positive\n",
        "Positive\n",
        "Negative\n",
        "Negative\n",
    ]
}
'''

'''
# 4-shot
data = {
    "text": [
        "Classify the following sentence as positive or negative:\n\nSentence: 'I love this movie.'\n Answer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'I do like the main characters.'\n Answer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'I hate this movie.'\n Answer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'How can this movie be awful like this?'\n Answer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'One of the film's most striking aspects is its exploration of the human psyche and the consequences of one's actions.' \nAnswer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'However, the film's dark tone and bleak outlook may not appeal to all audiences.' \nAnswer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'Viewers who prefer more straightforward narratives and happier endings may find the film's dark tone and complex storyline difficult to digest.' \nAnswer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'The movie is lush and beautiful , and the actors are well-chosen.' \nAnswer:",
    ],
    "label": [
        "Positive\n",
        "Positive\n",
        "Negative\n",
        "Negative\n",
        "Positive\n",
        "Negative\n",
        "Negative\n",
        "Positive\n",
    ]
}
'''

# reinforced 4-shot learning (data from trainset.head(10))
data = {
    "text": [
        "Classify the following sentence as positive or negative:\n\nSentence: 'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered \"controversial\" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it's not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films.<br /><br />I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn't have much of a plot. \
'\n Answer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'I used to be an avid viewer until I personally spent long cold hours helping build a home for the White Family, only to be sickened to see the house a year later. All of the beautiful rock landscaping has been removed, the gorgeous rock sidewalk and front fountain have been removed, all the pine trees and pecan trees in the front have been cut down, sprinkler system has been ripped out. It now looks like a disaster area. They don't even live there any more... they live \"in town\" and come out only for the weekend. It sickens me to think of all the hours that the great people of Oklahoma donated to these people and to see the result. The story that we all saw on TV wasn't completely the truth... don't believe every thing you see and hear. \
'\n Answer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'Maiden Voyage is just that. I'd like to say straight away that I watched 5mins of this before I just couldn't stand it anymore. As already stated in another comment, this film doesn't fall into the whole \"so bad it's good\" thing, it's just bad. The acting is awful, the sfx are poor, and the story is bland and stupid. Even the extras suck, the \"bag guy guards\" and such appear to hold their weapons like water pistols.<br /><br />Don't even bother watching this film, the only thing special about it is that, no matter how low your expectations are, you will still be disappointed. \
'\n Answer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'To be honest I knew what to expect before I watched this film, and I've got to say it has the worst acting I've ever seen. It does have its moments, and on a comedy level its very entertaining, but i'm afraid its not scary, and stupidity is taken to a new level. There's a lot of unnecessary gore, and the plot is all over the place. I have no idea why the aliens were evil, and why they even came to this remote part of wales, (i mean who'd go there anyway?) but I didn't care at that point, because I was amused by the costumes, and the bad CGI. As far as B-movies go, this deserves the title of 'being so bad, its good', and kudos to the film-makers, because they probably knew what they were doing. Long may these films continue..... \
'\n Answer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'Does anyone know what kind of pickup John T drove? I looks like a mid to late 70's Ford. This movie is my favorite as well as my wife's. It was the first memorable movie we saw as a married couple. The pick up is of interest as it is similar to the first truck I drove and recently found another like it. I would like to restore the pick up I have to resemble the on in the movie. Also the music was awesome, and the acting was great. Where and what is the lady who portrayed John's aunt? Also did John have a stunt double for the scene on the tower when he almost fell? Also what year of Mustang did Debra W drive in this show. It looked like a 60's model. Thanks, \
 \nAnswer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'Budget limitations, time restrictions, shooting a script and then cutting it, cutting it, cutting it... This crew is a group of good, young filmmakers; thoughtful in this script - yes, allegorical - clever in zero-dollar effects when time and knowledge is all you have, relying on actors and friends and kind others for their time, devotion, locations; and getting a first feature in the can, a 1-in-1000 thing. These guys make films. Good ones. Check out their shorts collection \"Heartland Horrors\" and see the development. And I can vouch, working with them is about the most fun thing you'll do in the business. I'm stymied by harsh, insulting criticism for this film, wondering if one reviewer even heard one word of dialogue, pondered one thought or concept, or if all that was desired of this work was the visual gore of bashing and slashing to satisfy some mindless view of what horror should mean to an audience. Let \"The Empty Acre\" bring itself to you. Don't preconceive what you expect it should be just because it gets put in the horror/thriller genre due to its supernatural premise. It's a drama with depth beyond how far you can stick a blade into someone with a reverence for a message that doesn't assault your brain's visual center, but rather, draws upon one's empathetic imagination to experience other's suffering of mind and spirit. mark ridgway, Curtis,\" The Empty Acre\" \
' \nAnswer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'After reading some quite negative views for this movie, I was not sure whether I should fork out some money to rent it. However, it was a pleasant surprise. I haven't seen the original movie, but if its better than this, I'd be in heaven.<br /><br />Tom Cruise gives a strong performance as the seemingly unstable David, convincing me that he is more than a smile on legs (for only the third time in his career- the other examples were Magnolia and Born on the Fourth of July). Penelope Cruz is slightly lightweight but fills the demands for her role, as does Diaz. The only disappointment is the slightly bland Kurt Russell. In the movie, however, it is not the acting that really impresses- its the filmmaking.<br /><br />Cameron Crowe excels in the director's role, providing himself with a welcome change of pace from his usual schtick. The increasing insanity of the movie is perfectly executed by Crowe (the brief sequence where Cruise walks through an empty Time Square is incredibly effective). The soundtrack (a distinguishing feature of a Crowe movie) is also sublime.<br /><br />You will be shocked and challenged as a viewer. The plot does seem a little contrived but the issues explored behind it are endlessly discussable. The movie isn't perfect, but its a welcome change of pace for Cruise and Crowe and for those raised on a diet of Hollywood gloss, should be a revelation. \
 \nAnswer:",
        "Classify the following sentence as positive or negative:\n\nSentence: 'My parents took me to this movie when I was nine years old. I have never forgotten it. I had never before seen anything as beautiful as Elizabeth Taylor. (She was twenty-two when she made Elephant Walk) Remember, I'm nine, so the feelings aren't sexual, I just couldn't see anything else on the screen. I just wanted to sit at her feet like a puppy and stare up at her. She has begun to show her age, (She's almost seventy-four) but I still believe her to be one of the most beautiful and breathtaking women to ever have lived.<br /><br />I have seen the movie several times since, and it is a sappy melodrama. What saves it is, of course, Miss Taylor's beauty, magnificent scenery, the very impressive elephant stampede, and a well-made point on human arrogance in the face of nature.<br /><br />All in all, a well-spent couple of hours watching the movie channel or a rented video. \
' \nAnswer:",
    ],
    "label": [
        "Negative\n",
        "Negative\n",
        "Negative\n",
        "Negative\n",
        "Positive\n",
        "Positive\n",
        "Positive\n",
        "Positive\n",
    ]
}


# prefix = [
#     "Classify the following sentence as Positive or Negative:\n\nSentence: ",
# ]


adapter_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)
'''
adapter_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

model_with_adapter = get_peft_model(model, adapter_config)

dataset = load_dataset("./stanford-imdb/plain_text")



prefix = "Classify the following sentence as Positive or Negative: \n\nSentence: "

testset = dataset["test"].shard(num_shards=125, index=0)  # means the first 200 samples
eval_dataset = []


def do_shot(data):
    prompted_message = []
    prompted_message.append("Now you should classify sentences as positive or negative , and here are some examples:")
    for text,label in zip(data["text"] , data["label"]):
        prompted_message.append(text + label)

    return "".join(prompted_message)

acc_count = 0
message_prompt = do_shot(data)
print(message_prompt)

def extract_label(output_text):
    if "Positive" in output_text or "positive" in output_text :
        return 1
    elif "Negative"  in output_text or "negative" in output_text :
        return 0
    else:
        return np.random.randint(0,2)

for i in range(200):
    eval_dataset.append(message_prompt + prefix +  testset["text"][i])


def eval_acc():
    count = 0
    for i in range(200):
        inputs = tokenizer(eval_dataset[i], return_tensors="pt",
                           padding='max_length', truncation=True, max_length=8000)
        outputs = model_with_adapter.generate(**inputs, max_length=2048 * 4)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text[-100::])
        print(extract_label(generated_text[-100::]))
        if (extract_label(generated_text[-100::]) == testset["label"][i]):
            count += 1

    return count / 200


print(eval_acc())



import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from datetime import datetime
import spacy
import os
from my_all_wrappers import memorize,wrap_result

def save_as_pickle(obj,filename):
    file = open(filename, 'wb')
    pickle.dump(obj, file)
    file.close()

def open_a_pickle(filename):
    file = open(filename, 'rb')
    obj=pickle.load(file)
    file.close()
    return obj


@memorize
def load_cases(concatStoriesOfaCase=False):
    """load all cases from the pickle file and returns a list of stories.
    each stroy is a string that includes a source's narration of the events"""
    # open a file, where you stored the pickled data
    file = open('cases.ttt', 'rb')
    # dump information to that file
    cases = pickle.load(file)
    # close the file
    file.close()
    if concatStoriesOfaCase==False:
        return cases
    else:
        return ["\n".join(case) for case in cases]

@memorize
def conacat_stories_of_a_case(case_no):
    """load all cases from the pickle file and returns a string 
    that includes all stories of a cses concatenated to one another"""
    c=load_cases()[case_no]
    complete_case=""
    for i,s in enumerate (c):
        complete_case+=s+"\n"
    return complete_case

def most_common_in_a_story(story,top=10, extended_stop_words=True):
    """
    story 0: [('saudi', 12), ('aramco', 12), ('computer', 11), ('kubecka', 9), ('world', 7), ('company', 7), ('hard', 6), ('drive', 6), ('one', 5), ('oil', 5), ('said', 5), ('hack', 4), ('supply', 4), ('office', 4), ('without', 3), ('back', 3), ('never', 3), ('cnnmoney', 3), ('started', 3), ('employee', 3)]
    """ 
    #stopwords
    from nltk.corpus import stopwords
    stops=stopwords.words('english')
    if extended_stop_words:
        more_stops="also said say computer system comair bc Integrated Case Management icm obamacare web site healthcare united saudi aramco delta airlines airline Britain Border Control Systems UoX new york stock exchange market nyse problem issue pm million know user would amin bc hydro u jpmorgan chase netflix amazon website site company business application uk http sony playstation psn department bmw connecteddrive app bank dbs sutter hospital store loblaw jan feb mar apr may jun jul aug sep oct nov dec monday tuesday wednesday thursday friday saturday sunday interviewee ya think also obama like source skytrain ulster target inc day one"
        stops.extend(more_stops.lower().split())
  
    lower_tokens = [w for w in word_tokenize(story.lower()) if w.isalpha()]
    # Retain alphabetic words: alpha_only
    alpha_only = [t for t in lower_tokens if t.isalpha()]

    # Remove all stop words: no_stops
    no_stops = [t for t in alpha_only if t not in stops]

    # Instantiate the WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    # Lemmatize all tokens into a new list: lemmatized
    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

    # Create the bag-of-words: bow
    bow = Counter(lemmatized)

    # Print the 10 most common tokens
    return bow.most_common(top)

@wrap_result('most commone words in stories and the case')
def most_common_in_a_case(case_no,top=10, extended_stop_words=True):
    """
    --------most commone words in stories and the case--------
    story 0: [('saudi', 12), ('aramco', 12), ('computer', 11), ('kubecka', 9), ('world', 7), ('company', 7), ('hard', 6), ('drive', 6), ('one', 5), ('oil', 5), ('said', 5), ('hack', 4), ('supply', 4), ('office', 4), ('without', 3), ('back', 3), ('never', 3), ('cnnmoney', 3), ('started', 3), ('employee', 3)]
    story 1: [('saudi', 5), ('aramco', 4), ('said', 4), ('virus', 3), ('oil', 3), ('online', 2), ('hit', 2), ('access', 2), ('group', 2), ('affected', 2), ('added', 2), ('company', 2), ('attack', 2), ('message', 2), ('system', 2), ('government', 2), ('last', 2), ('workstation', 1), ('computer', 1), ('back', 1)]
    ...
    story 8: [('attack', 14), ('saudi', 13), ('aramco', 13), ('malware', 10), ('network', 7), ('system', 7), ('said', 6), ('oil', 6), ('group', 6), ('service', 5), ('also', 5), ('assault', 5), ('day', 4), ('workstation', 4), ('firm', 4), ('production', 4), ('security', 4), ('country', 4), ('machine', 4), ('restored', 3)]
    
    complete story: [('aramco', 85), ('attack', 77), ('saudi', 74), ('said', 65), ('oil', 56), ('company', 52), ('computer', 48), ('security', 33), ('virus', 28), ('network', 28), ('world', 27), ('kubecka', 23), ('system', 23), ('country', 20), ('group', 19), ('cyber', 19), ('iran', 18), ('drive', 17), ('production', 16), ('could', 16)]

    """
    c=cases[case_no]
    return_str=""
    complete_case=""
    for i,s in enumerate (c):
        complete_case+=s+"\n"
        d=most_common_in_a_story(s,top,extended_stop_words)
        return_str+=f'story {i}: {d}\n\n'
    
    d=most_common_in_a_story(complete_case,top,extended_stop_words)
    return_str+=f'complete story: {d}\n'
    
    return return_str

@wrap_result('top unqiue words in the case as opposed to other cases')
def top_unique_words_case(case_no, no_records=20):
    """
    --------top unqiue words in the case as opposed to other cases--------
    aramco | saudi | oil | virus | kubecka | attack | iran | arabia | malware | researchers | shamoon | sword | cyber | production | countries | drives | cutting | crude | gas | justice | attacks | supplies | 30,000 | world | security | al-saadan | bahrain | hacktivists | workstations | network | atrocities | exploration | gulf | infected | panetta | iranian | hackers | computers | energy | employees | al-falih | assault | barrels | burning | destroy | flag | flame | leon | precaution | producer | 
    """
    cases_storied_joined=["".join(case) for case in cases] #each case is a list of stories
    # cases_storied_joined[2] is now a string
    # len(cases_storied_joined) returns the number of cases in the list
    tokenized_stories = [word_tokenize(case.lower()) for case in cases_storied_joined]  
    dictionary = Dictionary(tokenized_stories)
    corpus = [dictionary.doc2bow(case) for case in tokenized_stories]
    tfidf = TfidfModel(corpus)
    
    doc = corpus[case_no] 
    # Calculate the tfidf weights of doc: tfidf_weights
    tfidf_weights = tfidf[doc]
    # Sort the weights from highest to lowest: sorted_tfidf_weights
    sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

    # Print the top 20 weighted words
    return_str=""
    for term_id, weight in sorted_tfidf_weights[:no_records]:
        return_str+=f'{dictionary.get(term_id)} | ' #weight, 
    
    return return_str

@wrap_result('Spacy entities')
def spacy_entities_of_a_case(case_no):
    """
    example result:
    --------Spacy entities--------
    NORP(Nationalities or religious or political groups): {'Saudi': 32, 'Saudi Aramco': 16, 'American': 7, 'Iranian': 6, 'Islamic': 2, 'Arab': 2, 'Iranians': 2, "Saudi Aramco's": 1, 'Shiite': 1, 'accuses Sunni Muslim': 1, 'Shiites': 1, 'Windows': 1, 'Qatari': 1, 'Sunni Muslim': 1, "Shi'ite": 1, 'Sunni': 1, 'Syrian': 1})
    DATE(Absolute or relative dates or periods): {'Sunday': 4, 'Three years ago': 3, '15 August': 3, '2012': 2, 'Thursday': 2, 'Aug. 15': 2, '2010': 2, 'May': 2, 'last week': 2, 'August 15': 2, '1970s': 1, 'mid-2012': 1, 'Wednesday, Aug. 15, 2012': 1, '17 days': 1, 'September 2012 to January 2013': 1, 'Five months later': 1, 'Last week': 1, 'today': 1, 'November': 1, 'Sept. 10': 1, 'the days': 1, 'Aug. 15, 2012': 1, 'annual': 1, 'Between September 2012 to January 2013': 1, 'five months': 1, 'earlier this year': 1, 'a full month': 1, 'this week': 1, 'last month': 1, 'June': 1, 'Last May': 1, 'several years': 1, 'This year': 1, 'two weeks': 1, 'September': 1, 'More than two months': 1, 'this month': 1, 'several days': 1, 'mid-August': 1, 'the last few years': 1, 'April': 1, 'this year': 1, 'last year': 1, 'month': 1, 'the day': 1, 'Saturday, 10 days': 1, 'a few days': 1, 'August 15, 2012': 1, 'August 25, 2012,': 1, 'the Eid holidays': 1, 'Monday': 1, 'last weekend': 1})
    ORDINAL("first", "second", etc.): {'first': 5})
    CARDINAL(Numerals that do not fall under another type): {'one': 9, '30,000': 4, 'two': 4, '35,000': 2, '50,000': 2, 'thousands': 2, 'tens of thousands': 2, 'four': 2, 'about 30,000': 2, 'One': 1, 'About 30,000': 1, 'More than 30,000': 1, 'three-fifths': 1, 'Two': 1, 'half': 1, 'more than 55,000': 1, '24': 1, 'roughly a dozen': 1, 'at least one': 1, 'three': 1, 'the estimated 40,000': 1})
    TIME(Times smaller than a day): {'a matter of hours': 1, 'the morning of': 1, 'That morning': 1, 'just a few hours': 1, 'morning': 1, '11:08': 1, 'hours': 1, '11:08 a.m.': 1})
    ORG(Companies, agencies, institutions, etc.): {'Aramco': 15, 'the Cutting Sword of Justice': 5, "Saudi Aramco's": 4, 'Saudi Aramco': 4, 'CNNMoney': 3, 'Reuters': 3, 'Kubecka': 2, 'the Financial Times': 2, 'Cutting Sword of Justice': 2, 'Sony Pictures': 1, "Kubecka's": 1, 'Saudi Arabian Oil Co.': 1, 'the Interior Ministry': 1, 'the Organization of Petroleum Exporting Countries': 1, 'OPEC': 1, 'Panetta’s Warning\n': 1, 'Defense': 1, 'Circumstantial Evidence\nAramco': 1, 'BBC': 1, 'BLACK HAT USA': 1, 'Aramco, Kubecka': 1, 'Energy Aspects': 1, 'Shamoon': 1, 'Opec': 1, 'Islam': 1, 'Symantec': 1, 'The New York Times': 1, 'I.P.': 1, 'Capital One': 1, 'the United Nations': 1, 'the Center for Strategic and International Studies': 1, 'the National Security Council': 1, 'MBR': 1, 'Al-Falih': 1, 'Analysis Saudi Aramco': 1, 'Core': 1})
    PERCENT(Percentage, including "%"): {'10%': 1, '40 percent': 1, '10 percent': 1, '80 percent': 1})
    LOC(Non-GPE locations, mountain ranges, bodies of water): {'the Middle East': 4, 'Africa': 2, 'Europe': 2, 'Southeast Asia': 2, 'the Persian Gulf': 2, 'Kharg Island': 2, 'Earth': 1, 'Silicon Valley': 1, 'Persian Gulf': 1, 'Gulf': 1})
    PERSON(People, including fictional): {'Kubecka': 16, 'Flame': 4, 'al-Saadan': 3, 'Wiper': 3, 'Chris Kubecka': 2, 'Khalid al-Falih': 2, 'Abdullah al-Saadan': 2, 'Leon Panetta': 2, 'Rob Rachwald': 2, 'Master Boot Record': 2, 'Screens': 1, 'Al Saud': 1, 'Mansour Al-Turki': 1, 'al-Turki': 1, 'Al-Saadan': 1, 'Bloomberg': 1, 'Black Hat USA': 1, 'Continuous': 1, 'Amrita Sen': 1, 'Ali Naimi': 1, 'Mohammed al-Mady': 1, 'Pearl Harbor': 1, 'Lailat al Qadr': 1, 'Muhammad': 1, 'Leon E. Panetta': 1, 'Alex Wong': 1, 'James A. Lewis': 1, 'Richard A. Clarke': 1, 'Saudi Aramco': 1, 'Bashar al-Assad': 1, 'Rachwald': 1, 'Reem Shamseddine': 1, 'Angus McDowall': 1, 'Andrew Torchia': 1, 'Anna Willard': 1, 'Gary Crosse': 1, "Saudi Aramco's": 1, 'Al-Saud': 1, 'Seculert': 1, 'Khalid A Al-Falih': 1, 'Kaspersky Lab': 1, 'Spring': 1})
    GPE(Countries, cities, states): {'Iran': 18, 'Saudi Arabia': 10, 'Bahrain': 6, 'the United States': 5, 'U.S.': 4, 'US': 4, 'Syria': 3, 'Imperva': 3, 'Las Vegas': 2, 'Netherlands': 2, "Saudi Arabia's": 2, 'Tehran': 2, 'Israel': 2, 'Thailand': 1, 'London': 1, 'Qatar': 1, 'United States': 1, 'The United States': 1, 'Shamoon': 1, 'America': 1, 'Washington': 1, 'New York': 1, 'Britain': 1, 'Riyadh': 1, 'Yemen': 1, 'Lebanon': 1, 'Pastebin': 1})
    WORK_OF_ART(Titles of books, songs, etc.): {'Cutting Sword of Justice': 3, 'Saudi Aramco': 1, 'the Night of Power': 1, 'Koran': 1, 'Arabian Gulf': 1})
    QUANTITY(Measurements, as of weight or distance): {'9.5 million barrels': 1, '9.7 million barrels': 1, '12.5 million barrels': 1, '9.4 million barrels': 1})
    FAC(Buildings, airports, highways, bridges, etc.): {'Shamoon': 4, 'Dhahran': 1})
    PRODUCT(Objects, vehicles, foods, etc. (not services)): {'Sabic': 1, 'BB&T.': 1})
    MONEY(Monetary values, including unit): {'100 per cent': 1, 'three-quarters': 1})
    LANGUAGE(Any named language): {'English': 1})
    """
    story=conacat_stories_of_a_case(case_no)
    dict_ent={} #dictionary of sets

    # Instantiate the English model: nlp
    nlp = spacy.load('en_core_web_sm',disable=['tagger', 'parser', 'matcher'])

    # Create a new document: doc
    doc = nlp(story)

    # Print all of the found entities and their labels
    for ent in doc.ents:
        if not (ent.label_ in dict_ent.keys()):
            dict_ent[ent.label_]=list()
        dict_ent[ent.label_].append(ent.text) 
    
    return_str=""
    for k,v in dict_ent.items():
        v=str(Counter(v)).replace('Counter(','',1)
        return_str+=f'{k}({spacy.explain(k)}): {v}\n\n'
    
    return return_str

@wrap_result('Spacy tokens & tags')
def spacy_token_tags_of_a_case(case_no):
    """
    token._pos(explanation of token._pos): {ordered list of instances using Counter}
    example:
    --------Spacy tokens & tags--------
    PROPN(proper noun): {'Aramco': 83, 'Saudi': 52, 'Kubecka': 24, 'Iran': 18, 'Arabia': 12, 'al': 11, 'Sword': 10, 'Justice': 10, 'August': 8, 'Cutting': 8, 'Shamoon': 8, 'United': 8, 'States': 7, 'Al': 6, 'IT': 6, 'Saadan': 6, 'Bahrain': 6, 'Gulf': 5, 'Panetta': 5, 'cyber': 5, 'Aug.': 4, 'Middle': 4, 'East': 4, 'Falih': 4, 'U.S.': 4, 'Leon': 4, 'Stuxnet': 4, 'US': 4, 'Flame': 4, 'Sunday': 4, 'CNNMoney': 3, 'September': 3, 'Khalid': 3, 'Persian': 3, 'Times': 3, 'Syria': 3, 'Wiper': 3, 'May': 3, 'Reuters': 3, 'Rachwald': 3, 'Imperva': 3, 'Chris': 2, 'Thursday': 2, 'Black': 2, 'Hat': 2, 'Las': 2, 'Vegas': 2, 'Ramadan': 2, 'Saud': 2, 'Netherlands': 2, 'Africa': 2, 'Europe': 2, 'Southeast': 2, 'Asia': 2, 'January': 2, 'Arabian': 2, 'Abdullah': 2, 'Turki': 2, 'International': 2, 'Armageddon': 2, 'USA': 2, 'Windows': 2, 'RasGas': 2, 'Tehran': 2, 'Financial': 2, 'Symantec': 2, 'Kharg': 2, 'Island': 2, 'New': 2, 'York': 2, 'Israel': 2, 'Iranians': 2, 'A.': 2, 'Rob': 2, 'Master': 2, 'Boot': 2, 'Record': 2, 'Earth': 1, 'Sony': 1, 'Pictures': 1, 'Wednesday': 1, 'Screens': 1, 'Thailand': 1, 'Remote': 1, 'Oil': 1, 'Co.': 1, 'Dhahran': 1, 'Major': 1, 'General': 1, 'Mansour': 1, 'Interior': 1, 'Ministry': 1, 'Organization': 1, 'Petroleum': 1, 'Exporting': 1, 'Countries': 1, 'Shiite': 1, 'OPEC': 1, 'Sunni': 1, 'Muslim': 1, 'Shiites': 1, 'Warning': 1, 'Defense': 1, 'Secretary': 1, 'war': 1, 'November': 1, 'Bloomberg': 1, 'Circumstantial': 1, 'Evidence': 1, 'Sept.': 1, 'Government': 1, 'BBC': 1, 'BLACK': 1, 'HAT': 1, 'Distrack': 1, 'Internet': 1, 'international': 1, 'Time': 1, 'Amrita': 1, 'Sen': 1, 'Energy': 1, 'Aspects': 1, 'London': 1, 'Qatar': 1, 'Opec': 1, 'Ali': 1, 'Naimi': 1, 'Mohammed': 1, 'Mady': 1, 'Sabic': 1, 'Pearl': 1, 'Harbor': 1, 'Cyber': 1, 'Islam': 1, 'Lailat': 1, 'Qadr': 1, 'Power': 1, 'Koran': 1, 'Muhammad': 1, 'Hand': 1, 'CaseOCT': 1, '.': 1, 'E.': 1, 'Photo': 1, 'Credit': 1, 'Alex': 1, 'Wong': 1, 'Getty': 1, 'virus': 1, 'Silicon': 1, 'Valley': 1, 'June': 1, 'Google': 1, 'I.P.': 1, 'America': 1, 'Capital': 1, 'One': 1, 'Washington': 1, 'Nations': 1, 'James': 1, 'Lewis': 1, 'Center': 1, 'Strategic': 1, 'Studies': 1, 'Richard': 1, 'Clarke': 1, 'National': 1, 'Security': 1, 'Council': 1, 'mid': 1, '-': 1, 'CEO': 1, 'April': 1, 'ministry': 1, 'national': 1, 'Britain': 1, "Shi'ite": 1, 'Riyadh': 1, 'President': 1, 'Bashar': 1, 'Assad': 1, 'MBR': 1, 'Reem': 1, 'Shamseddine': 1, 'Angus': 1, 'McDowall': 1, 'Andrew': 1, 'Torchia': 1, 'Anna': 1, 'Willard': 1, 'Gary': 1, 'Crosse': 1, 'Analysis': 1, 'Saturday': 1, 'Eid': 1, 'Yemen': 1, 'Lebanon': 1, 'Egypt': 1, 'Seculert': 1, 'Core': 1, 'A': 1, 'Pastebin': 1, 'Monday': 1, 'DDoS': 1, 'Kaspersky': 1, 'Lab': 1})
    VERB(verb): {'said': 65, 'affected': 10, 'had': 9, 'claimed': 9, 'hit': 9, 'called': 9, 'stop': 7, 'told': 6, 'say': 6, 'wiped': 5, 'known': 5, 'restored': 5, 'used': 5, 'do': 5, 'destroyed': 4, 'started': 4, 'began': 4, 'prevent': 4, 'according': 4, 'bought': 4, 'took': 4, 'isolated': 4, 'taken': 4, 'following': 4, 'named': 4, 'targeted': 4, 'owned': 4, 'including': 4, 'taking': 4, 'set': 4, 'burning': 4, 'destroy': 4, 'seen': 3, 'pay': 3, 'calling': 3, 'remained': 3, 'forced': 3, 'gone': 3, 'paid': 3, 'cut': 3, 'brought': 3, 'know': 3, 'restricted': 3, 'added': 3, 'blamed': 3, 'run': 3, 'involved': 3, 'based': 3, 'aimed': 3, 'compromised': 3, 'originated': 3, 'led': 3, 'has': 3, 'disrupted': 3, 'recover': 3, 'attack': 3, 'sent': 3, 'making': 3, 'suspected': 3, 'believe': 3, 'infected': 3, 'using': 2, 'heard': 2, 'clicked': 2, 'noticed': 2, 'shut': 2, 'Cutting': 2, 'citing': 2, 'support': 2, 'goes': 2, 'spreading': 2, 'automated': 2, 'wrote': 2, 'stopped': 2, 'living': 2, 'help': 2, 'secure': 2, 'flew': 2, 'halting': 2, 'became': 2, 'bankrupted': 2, 'identified': 2, 'struck': 2, 'apologising': 2, 'reads': 2, 'repaired': 2, 'intrude': 2, 'reported': 2, 'phishing': 2, 'declined': 2, 'failed': 2, 'supplies': 2, 'erased': 2, 'estimated': 2, 'operating': 2, 'arrived': 2, 'attacked': 2, 'face': 2, 'turn': 2, 'tracking': 2, 'figure': 2, 'meant': 2, 'needed': 2, 'targeting': 2, 'take': 2, 'build': 2, 'gave': 2, 'become': 2, 'have': 2, 'trying': 2, 'made': 2, 'work': 2, 'buying': 2, 'came': 2, 'revealed': 2, 'disclosed': 2, 'growing': 2, 'are': 2, 'protect': 2, 'want': 2, 'warned': 2, 'name': 2, 'continues': 2, 'initiate': 2, 'replacing': 2, 'found': 2, 'analyzing': 2, 'designed': 2, 'discovered': 2, 'written': 2, 'intended': 2, 'inserted': 2, 'threatened': 2, 'carried': 2, 'put': 2, 'comment': 2, 'operate': 2, 'posting': 2, 'confirmed': 2, 'hacked': 2, 'use': 2, 'suffered': 1, 'witnessed': 1, 'learning': 1, 'seeking': 1, 'turned': 1, 'supply': 1, 'propelled': 1, 'comes': 1, 'felt': 1, 'spoke': 1, 'asked': 1, 'confirm': 1, 'respond': 1, 'duped': 1, 'recalled': 1, 'opened': 1, 'acting': 1, 'flickering': 1, 'disappear': 1, 'ripped': 1, 'viewed': 1, 'pumping': 1, 'explained': 1, 'Managing': 1, 'happen': 1, 'passed': 1, 'needing': 1, 'faxed': 1, 'selling': 1, 'relented': 1, 'giving': 1, 'keep': 1, 'flowing': 1, 'hired': 1, 'flexed': 1, 'purchase': 1, 'backed': 1, 'secured': 1, 'expanded': 1, 'caught': 1, 'carries': 1, 'disrupting': 1, 'raising': 1, 'pose': 1, 'identify': 1, 'reach': 1, 'accused': 1, 'interfering': 1, 'proven': 1, 'denies': 1, 'accuses': 1, 'discriminating': 1, 'suggested': 1, 'improving': 1, 'target': 1, 'purged': 1, 'pumped': 1, 'compiled': 1, 'damaged': 1, 'implicating': 1, 'fit': 1, 'increased': 1, 'faced': 1, 'continue': 1, 'enhance': 1, 'claims': 1, 'maintains': 1, 'displaying': 1, 'blames': 1, 'describes': 1, 'rocked': 1, 'averted': 1, 'mounted': 1, 'disappeared': 1, 'fail': 1, 'lasted': 1, 'disconnected': 1, 'travelling': 1, 'isolating': 1, 'Imagine': 1, 'went': 1, 'managing': 1, 'handling': 1, 'go': 1, 'buy': 1, 'were': 1, 'invested': 1, 'securing': 1, 'crippled': 1, 'pwned': 1, 'devoted': 1, 'remain': 1, 'got': 1, 'assembled': 1, 'staffed': 1, 'expand': 1, 'complemented': 1, 'needs': 1, 'need': 1, 'think': 1, 'considering': 1, 'recruit': 1, 'cutting': 1, 'assembling': 1, 'emphasized': 1, 'affect': 1, 'change': 1, 'pave': 1, 'knowing': 1, 'rival': 1, 'utilized': 1, 'fly': 1, 'get': 1, 'driving': 1, 'reused': 1, 'rebuilt': 1, 'decided': 1, 'figuring': 1, 'consuming': 1, 'knowcked': 1, 'showed': 1, 'reover': 1, 'highlighting': 1, 'tried': 1, 'bring': 1, 'succeeding': 1, 'penetrating': 1, 'raise': 1, 'accounts': 1, 'holds': 1, 'rule': 1, 'liquefied': 1, 'come': 1, 'producing': 1, 'achieve': 1, 'allow': 1, 'related': 1, 'tightened': 1, 'is': 1, 'protected': 1, 'slow': 1, 'apply': 1, 'pointed': 1, 'escalating': 1, 'cyber': 1, 'fed': 1, 'repeated': 1, 'spread': 1, 'picked': 1, 'knew': 1, 'inflict': 1, 'stayed': 1, 'prepare': 1, 'celebrating': 1, 'unleashed': 1, 'regarded': 1, 'Keeping': 1, 'offered': 1, 'cited': 1, 'upset': 1, 'looked': 1, 'disabling': 1, 'disturbing': 1, 'segregated': 1, 'assured': 1, 'spilled': 1, 'authorized': 1, 'speak': 1, 'embedded': 1, 'replace': 1, 'report': 1, 'included': 1, 'erasing': 1, 'noted': 1, 'given': 1, 'raised': 1, 'fired': 1, 'maintained': 1, 'siphoning': 1, 'commissioned': 1, 'misdirect': 1, 'refer': 1, 'sue': 1, 'removing': 1, 'posted': 1, 'grab': 1, 'stated': 1, 'blame': 1, 'engineered': 1, 'demonstrates': 1, 'developing': 1, 'bolder': 1, 'expected': 1, 'going': 1, 'deal': 1, 'gain': 1, 'decide': 1, 'access': 1, 'wake': 1, 'proved': 1, 'mess': 1, 'expect': 1, 'resumed': 1, 'announced': 1, 'cleansed': 1, 'like': 1, 'emphasize': 1, 'assure': 1, 'functioning': 1, 'continued': 1, 'bounce': 1, 'continuing': 1, 'elaborate': 1, 'conducted': 1, 'make': 1, 'disrupt': 1, 'focused': 1, 'disputed': 1, 'forcing': 1, 'disconnect': 1, 'handles': 1, 'attributed': 1, 'built': 1, 'try': 1, 'completing': 1, 'signed': 1, 'launched': 1, 'back': 1, 'supporting': 1, 'contacted': 1, 'achieved': 1, 'corrupts': 1, 'overwrites': 1, 'render': 1, 'ensure': 1, 'reinforce': 1, 'editing': 1, 'floored': 1, 'resulted': 1, 'suspend': 1, 'suspended': 1, 'pledged': 1, 'investigate': 1, 'promised': 1, 'improve': 1, 'guard': 1, 'impacted': 1, 'cleaned': 1, 'returned': 1, 'resuming': 1, 'compromising': 1, 'implanting': 1, 'lending': 1, 'perpetrator': 1, 'featured': 1, 'implicated': 1, 'emerged': 1, 'write': 1, 'replaced': 1, 'According': 1, 'extract': 1, 'uploading': 1, 'supposed': 1, 'uploaded': 1, 'follow': 1, 'described': 1, 'distributed': 1, 'clog': 1, 'jump': 1, 'wiping': 1, 'prospecting': 1, 'sponsored': 1, 'seems': 1, 'view': 1, 'emerges': 1, 'motivated': 1, 'hitting': 1, 'ruling': 1, 'putting': 1})
    DET(determiner): {'the': 355, 'a': 131, 'The': 48, 'an': 25, 'this': 12, 'all': 10, 'A': 8, 'that': 5, 'no': 5, 'some': 4, 'That': 3, 'every': 3, 'An': 3, 'Every': 2, 'any': 2, 'those': 2, 'Some': 1, 'No': 1, 'This': 1, 'Both': 1, 'these': 1})
    ADJ(adjective): {'Saudi': 22, 'hard': 16, 'largest': 10, 'several': 10, 'other': 9, 'online': 9, 'last': 9, 'internal': 9, 'American': 7, 'corporate': 7, 'global': 7, 'cyber': 7, 'first': 6, 'offline': 6, 'Iranian': 6, 'main': 6, 'new': 5, 'former': 5, 'most': 5, 'such': 5, 'same': 5, 'few': 4, 'higher': 4, 'crude': 4, 'unaffected': 4, 'nuclear': 4, 'valuable': 3, 'royal': 3, 'electronic': 3, 'outside': 3, 'Most': 3, 'chief': 3, 'Arab': 3, 'further': 3, 'similar': 3, 'remote': 3, 'national': 3, 'more': 3, 'private': 3, 'destructive': 3, 'significant': 3, 'responsible': 3, 'worst': 2, 'sheer': 2, 'recent': 2, 'bad': 2, 'Islamic': 2, 'holy': 2, 'unplugged': 2, 'domestic': 2, 'independent': 2, 'massive': 2, 'smaller': 2, 'least': 2, 'early': 2, 'precautionary': 2, 'sudden': 2, 'key': 2, 'illegal': 2, 'Last': 2, 'More': 2, 'Aramco': 2, 'sophisticated': 2, 'secure': 2, 'possible': 2, 'different': 2, 'good': 2, 'Corporate': 2, 'available': 2, 'real': 2, 'local': 2, 'international': 2, 'natural': 2, 'worse': 2, 'protective': 2, 'biggest': 2, 'various': 2, 'privileged': 2, 'able': 2, 'isolated': 2, 'external': 2, 'Sunni': 2, 'unknown': 2, 'infected': 2, 'hacktivist': 2, 'monstrous': 1, 'average': 1, 'mysterious': 1, 'little': 1, 'actual': 1, 'weird': 1, 'authoritarian': 1, 'criminal': 1, 'frantic': 1, 'steady': 1, 'dead': 1, 'interoffice': 1, 'Lengthy': 1, 'lucrative': 1, 'free': 1, 'fell': 1, 'constrained': 1, 'front': 1, 'unidentified': 1, 'eastern': 1, 'foreign': 1, 'ultimate': 1, 'fellow': 1, 'individual': 1, 'complicit': 1, 'refined': 1, 'single': 1, 'safe': 1, 'entire': 1, 'small': 1, 'relative': 1, 'circumstantial': 1, 'Former': 1, 'wrong': 1, 'modern': 1, 'old': 1, 'industrial': 1, 'pronged': 1, 'great': 1, 'religious': 1, 'best': 1, 'Continuous': 1, 'proactive': 1, 'successful': 1, 'grey': 1, 'expensive': 1, 'better': 1, 'annual': 1, 'whole': 1, 'unique': 1, 'usable': 1, 'fastest': 1, 'giant': 1, 'devastating': 1, 'full': 1, 'certain': 1, 'weak': 1, 'likely': 1, 'high': 1, 'spare': 1, 'critical': 1, 'intricate': 1, 'unstable': 1, 'Other': 1, 'crucial': 1, 'much': 1, 'holiest': 1, 'specific': 1, 'red': 1, 'close': 1, 'exact': 1, 'erasing': 1, 'upper': 1, 'correct': 1, 'disconnected': 1, 'inappropriate': 1, 'subsequent': 1, 'Qatari': 1, 'Multiple': 1, 'hostile': 1, 'militant': 1, 'political': 1, 'economic': 1, 'Current': 1, 'complex': 1, 'English': 1, 'Muslim': 1, 'Syrian': 1, 'strong': 1, 'minded': 1, 'accomplish': 1, 'Symantec': 1, 'unusable': 1, 'unusual': 1, 'typical': 1, 'only': 1, 'Additional': 1, 'fresh': 1, 'malicious': 1, 'normal': 1, 'primary': 1, 'operational': 1, 'impossible': 1, 'latest': 1, 'due': 1, 'original': 1, 'fuelled': 1, 'dangerous': 1, 'Similar': 1, 'previous': 1, 'wise': 1, 'motivated': 1, 'clearer': 1})
    NOUN(noun): {'attack': 60, 'oil': 50, 'company': 44, 'security': 32, 'computer': 29, 'virus': 27, 'world': 26, 'network': 26, 'systems': 21, 'computers': 18, 'group': 18, 'countries': 16, 'production': 16, 'malware': 16, 'attacks': 14, 'drives': 12, 'officials': 12, 'employees': 11, 'data': 11, 'access': 11, 'time': 10, 'team': 10, 'researchers': 10, 'hackers': 9, 'supplies': 9, 'state': 9, 'experts': 9, 'companies': 8, 'government': 7, 'firm': 7, 'responsibility': 7, 'gas': 7, 'days': 7, 'energy': 7, 'code': 7, 'month': 6, 'drive': 6, 'industry': 6, 'investigation': 6, 'cyber': 6, 'machines': 6, 'year': 6, 'workstations': 6, 'years': 5, 'email': 5, 'office': 5, 'Internet': 5, 'Oil': 5, 'day': 5, 'business': 5, 'governments': 5, 'mail': 5, 'people': 5, 'giant': 5, 'message': 5, 'crimes': 5, 'atrocities': 5, '-': 5, 'files': 5, 'exploration': 5, 'hacktivists': 5, 'hack': 4, 'cyberattack': 4, 'technology': 4, 'information': 4, 'barrels': 4, 'prices': 4, 'cybersecurity': 4, 'producer': 4, 'precaution': 4, 'website': 4, 'damage': 4, 'week': 4, 'kingdom': 4, 'threat': 4, 'e': 4, 'intelligence': 4, 'thousands': 4, 'recovery': 4, 'attackers': 4, 'operations': 4, 'place': 4, 'flag': 4, 'name': 4, 'statement': 4, 'assault': 4, 'hours': 3, 'tank': 3, 'trucks': 3, 'typewriters': 3, 'cyberattacks': 3, 'advisor': 3, 'hacking': 3, 'conference': 3, 'morning': 3, 'support': 3, 'family': 3, 'regime': 3, 'country': 3, 'servers': 3, 'rest': 3, 'partners': 3, 'corporation': 3, 'months': 3, 'size': 3, 'source': 3, 'income': 3, 'president': 3, 'crude': 3, 'percent': 3, 'evidence': 3, 'markets': 3, 'capacity': 3, 'facilities': 3, 'style': 3, 'effort': 3, 'control': 3, 'things': 3, 'professionals': 3, 'nations': 3, 'target': 3, 'executives': 3, 'internet': 3, 'secretary': 3, 'work': 3, 'sabotage': 3, 'PCs': 3, 'image': 3, 'retaliation': 3, 'service': 3, 'blog': 3, 'services': 3, 'way': 2, 'gasoline': 2, 'refills': 2, 'ability': 2, 'risk': 2, 'person': 2, 'comment': 2, 'technicians': 2, 'link': 2, 'warning': 2, 'centers': 2, 'contracts': 2, 'paper': 2, 'phones': 2, 'Employees': 2, 'page': 2, 'consultant': 2, 'satellite': 2, 'offices': 2, 'factory': 2, 'floors': 2, 'line': 2, 'price': 2, 'system': 2, 'workstation': 2, 'inconvenience': 2, 'measure': 2, 'disruption': 2, 'sectors': 2, 'forum': 2, 'attempt': 2, 'executive': 2, 'exporter': 2, 'spear': 2, 'concerns': 2, 'output': 2, 'vice': 2, 'planning': 2, 'flow': 2, 'rulers': 2, 'drop': 2, 'breach': 2, 'warfare': 2, 'tens': 2, 'staff': 2, 'fax': 2, 'supply': 2, 'resources': 2, 'employee': 2, 'center': 2, 'date': 2, 'understanding': 2, 'program': 2, 'guys': 2, 'culture': 2, 'example': 2, 'method': 2, 'hydrocarbon': 2, 'points': 2, 'cent': 2, 'claim': 2, 'defense': 2, 'escalation': 2, 'case': 2, 'communications': 2, 'word': 2, 'addresses': 2, 'list': 2, 'memory': 2, 'Shamoon': 2, 'ministry': 2, 'software': 2, 'exports': 2, 'programmers': 2, 'names': 2, 'capability': 2, 'networks': 2, 'core': 2, 'distribution': 2, 'sources': 2, 'director': 2, 'organization': 2, 'result': 2, 'claims': 2, 'history': 1, 'details': 1, 'matter': 1, '%': 1, 'faxes': 1, 'cost': 1, 'pale': 1, 'comparison': 1, 'reverberations': 1, 'experience': 1, 'tale': 1, 'presentation': 1, 'account': 1, 'request': 1, 'mid-2012': 1, 'scam': 1, 'holiday': 1, 'Files': 1, 'explanation': 1, 'tyrants': 1, 'disasters': 1, 'injustice': 1, 'oppression': 1, 'rush': 1, 'cables': 1, 'backs': 1, 'records': 1, 'Drilling': 1, 'turmoil': 1, 'shipping': 1, 'Office': 1, 'reports': 1, 'Contracts': 1, 'deals': 1, 'signatures': 1, 'army': 1, 'life': 1, 'muscle': 1, 'representatives': 1, 'manufacturing': 1, 'swoop': 1, 'World': 1, 'flooding': 1, 'today': 1, 'news': 1, 'city': 1, 'spokesman': 1, 'progress': 1, 'goal': 1, '’s': 1, 'member': 1, 'affairs': 1, 'fifths': 1, 'reserves': 1, 'charge': 1, 'brewing': 1, 'steps': 1, 'breaches': 1, 'none': 1, 'contractors': 1, 'effect': 1, 'products': 1, 'fact': 1, 'assurance': 1, 'percentage': 1, 'interviews': 1, 'pattern': 1, 'activity': 1, 'uranium': 1, 'enrichment': 1, 'future': 1, 'measures': 1, 'attendees': 1, 'operation': 1, 'job': 1, 'emails': 1, 'drilling': 1, 'pumping': 1, 'school': 1, 'shipment': 1, 'shutdown': 1, 'payment': 1, 'miles': 1, 'irony': 1, 'desktops': 1, 'forensics': 1, 'mystery': 1, 'half': 1, 'IT': 1, 'teams': 1, 'observances': 1, 'investigators': 1, 'part': 1, 'monitoring': 1, 'environment': 1, 'set': 1, 'skills': 1, 'tinge': 1, 'evil': 1, 'scratch': 1, 'costs': 1, 'kind': 1, 'collaboration': 1, 'culuture': 1, 'decisions': 1, 'Humans': 1, 'awareness': 1, 'help': 1, 'revenues': 1, 'economies': 1, 'advantage': 1, 'fleet': 1, 'airplanes': 1, 'shipments': 1, 'buyers': 1, 'essence': 1, 'challenge': 1, 'impact': 1, 'desktop': 1, 'aim': 1, 'media': 1, 'traders': 1, 'diplomats': 1, 'confirmation': 1, 'alarms': 1, 'piece': 1, 'consultancy': 1, 'sign': 1, 'situation': 1, 'producers': 1, 'wave': 1, 'tension': 1, 'programme': 1, 'strains': 1, 'centre': 1, 'meeting': 1, 'cartel': 1, 'minister': 1, 'aims': 1, 'devices': 1, 'petrochemical': 1, 'maker': 1, 'value': 1, 'interview': 1, 'area': 1, 'limit': 1, 'defence': 1, 'posts': 1, 'accusation': 1, 'continents': 1, 'Officials': 1, 'nights': 1, 'Night': 1, 'revelation': 1, 'acts': 1, 'quarters': 1, 'documents': 1, 'spreadsheets': 1, 'mails': 1, 'Cash': 1, 'perpetrator': 1, 'speech': 1, 'dangers': 1, 'activists': 1, 'policies': 1, 'Images': 1, 'herrings': 1, 'examination': 1, 'parties': 1, 'dozen': 1, 'specialists': 1, 'handle': 1, 'sample': 1, 'bragging': 1, 'sorts': 1, 'kill': 1, 'switch': 1, 'timer': 1, 'creators': 1, 'mechanism': 1, 'Computer': 1, 'component': 1, 'light': 1, 'connections': 1, 'rigs': 1, 'terminal': 1, 'conduit': 1, 'suspicions': 1, 'shots': 1, 'war': 1, 'hand': 1, 'centrifuges': 1, 'facility': 1, 'Security': 1, 'clues': 1, 'blame': 1, 'body': 1, 'water': 1, 'maps': 1, 'event': 1, 'insider': 1, 'insiders': 1, 'USB': 1, 'stick': 1, 'PC': 1, 'blocks': 1, 'proof': 1, 'Researchers': 1, 'contractor': 1, 'rumor': 1, 'speculation': 1, 'weeks': 1, 'banks': 1, 'banking': 1, 'Web': 1, 'sites': 1, 'requests': 1, 'interests': 1, 'mission': 1, 'response': 1, 'finger': 1, 'pointing': 1, 'concern': 1, 'skill': 1, 'diplomat': 1, 'expert': 1, 'sides': 1, 'dance': 1, 'fight': 1, 'aftermath': 1, 'call': 1, 'lot': 1, 'counterterrorism': 1, 'official': 1, 'lots': 1, 'targets': 1, 'thing': 1, 'stakeholders': 1, 'customers': 1, 'businesses': 1, 'wellhead': 1, 'websites': 1, 'Emails': 1, 'causes': 1, 'incident': 1, 'Information': 1, 'infrastructure': 1, 'groups': 1, 'sanctions': 1, 'worm': 1, 'weapons': 1, 'POSTING': 1, 'language': 1, 'posting': 1, 'bulletin': 1, 'board': 1, 'troops': 1, 'protesters': 1, 'rebels': 1, 'milestone': 1, 'hobbyists': 1, 'developers': 1, 'results': 1, 'sector': 1, 'W32.Disttrack': 1, 'Threats': 1, 'payloads': 1, 'attempts': 1, 'means': 1, 'recurrence': 1, 'type': 1, 'reporting': 1, 'desking': 1, 'outbreak': 1, 'decision': 1, 'period': 1, 'meantime': 1, 'assaults': 1, 'holidays': 1, 'enterprise': 1, 'Production': 1, 'plants': 1, 'neighboring': 1, 'number': 1, 'credibility': 1, 'hacker': 1, 'victim': 1, 'analysis': 1, 'boot': 1, 'router': 1, 'admin': 1, 'passwords': 1, 'address': 1, 'password': 1, 'exec': 1, 'leak': 1, 'weekend': 1, 'fruits': 1, 'strategy': 1, 'past': 1, 'application': 1, 'denial': 1, 'traffic': 1, 'use': 1, 'Hacktivists': 1, 'trend': 1, 'post': 1, 'discovery': 1, 'espionage': 1, 'tool': 1, 'firms': 1, 'hacktivism': 1, 'skepticism': 1, 'picture': 1, 'Spring': 1, 'revolts': 1, 'motives': 1})
    ADP(adposition): {'of': 179, 'in': 96, 'to': 55, 'on': 52, 'for': 39, 'at': 32, 'by': 32, 'from': 29, 'with': 22, 'as': 19, 'after': 12, 'up': 10, 'about': 9, 'In': 8, 'into': 8, 'down': 8, 'against': 8, 'over': 7, 'out': 6, 'than': 6, 'off': 5, 'around': 4, 'through': 4, 'before': 4, 'during': 3, 'On': 3, 'per': 3, 'within': 3, 'Without': 2, 'After': 2, 'As': 2, 'amid': 2, 'among': 2, 'inside': 2, 'Until': 1, 'without': 1, 'like': 1, 'outside': 1, 'For': 1, 'Between': 1, 'between': 1, 'behind': 1, 'By': 1, 'Within': 1, 'Before': 1, 'along': 1})
    NUM(numeral): {'one': 13, '15': 10, '2012': 8, '30,000': 8, 'two': 6, '10': 5, 'million': 4, 'Three': 3, 'three': 3, '35,000': 2, '50,000': 2, '2013': 2, '2010': 2, 'four': 2, '11:08': 2, '1970s': 1, 'One': 1, '9.5': 1, '17': 1, 'Five': 1, '40': 1, '9.7': 1, '12.5': 1, 'Two': 1, '9.4': 1, 'five': 1, '100': 1, '55,000': 1, '’s': 1, '24': 1, '80': 1, '25': 1, '40,000': 1})
    PUNCT(punctuation): {',': 301, '.': 262, '-': 52, '"': 50, '“': 32, '”': 31, '--': 13, '—': 12, '(': 5, ')': 5, ':': 4, '–': 2, ';': 2, '‘': 1, '’': 1, ']': 1})
    SPACE(space): {'\n': 115, '\n\n': 23, '\n\n\n': 5, '\n\n\n\n': 3, '\n\n\n\n\n\n': 1, '\n\n\n\n\n': 1, 'U.S.-based': 1})
    ADV(adverb): {'also': 10, 'back': 9, 'now': 8, 'most': 6, 'online': 6, 'ago': 4, 'never': 4, 'temporarily': 4, 'still': 4, 'then': 4, 'only': 4, 'away': 3, 'just': 3, 'further': 3, 'more': 3, 'here': 3, 'long': 3, 'very': 3, 'ever': 2, 'partially': 2, 'totally': 2, 'publicly': 2, 'ahead': 2, 'physically': 2, 'directly': 2, 'already': 2, 'even': 2, 'slightly': 2, 'easily': 2, 'so': 2, 'home': 2, 'instead': 2, 'together': 2, 'Immediately': 2, 'much': 2, 'about': 2, 'However': 2, 'previously': 2, 'suddenly': 1, 'sometime': 1, 'all': 1, 'currently': 1, 'else': 1, 'later': 1, 'newly': 1, 'About': 1, 'normally': 1, 'largely': 1, 'before': 1, 'around': 1, 'yet': 1, 'swiftly': 1, 'first': 1, 'immediately': 1, 'aka': 1, 'heavily': 1, 'tremendously': 1, 'right': 1, 'too': 1, 'earlier': 1, 'initially': 1, 'Of': 1, 'course': 1, 'however': 1, 'Just': 1, 'probably': 1, 'roughly': 1, 'a.m.': 1, 'mainly': 1, 'intermittently': 1, 'faster': 1, 'Still': 1, 'no': 1, 'longer': 1, 'remotely': 1, 'as': 1, 'reliably': 1, 'widely': 1, 'allegedly': 1, 'at': 1, 'least': 1, 'presumably': 1, 'fully': 1, 'especially': 1, 'thus': 1, 'Over': 1, 'reportedly': 1, 'rather': 1, 'typically': 1, 'rarely': 1, 'politically': 1, 'solely': 1})
    CCONJ(coordinating conjunction): {'and': 115, 'or': 17, 'but': 15, 'But': 7, 'nor': 4, 'And': 2, 'Neither': 1, 'both': 1})
    PRON(pronoun): {'it': 39, 'its': 32, 'that': 27, 'which': 26, 'It': 19, 'they': 12, 'our': 10, 'we': 7, 'their': 7, 'who': 7, 'all': 6, 'she': 6, 'you': 6, 'this': 4, 'We': 4, 'he': 4, 'itself': 3, 'This': 3, 'I': 3, 'IT': 3, 'what': 3, 'them': 2, 'her': 2, 'everyone': 2, 'those': 2, 'There': 2, 'there': 2, 'She': 1, 'Somebody': 1, 'anything': 1, 'my': 1, 'Everyone': 1, 'any': 1, 'You': 1, 'Her': 1, 'something': 1, 'everything': 1, 'nothing': 1, 'Everything': 1, 'both': 1, 'He': 1, 'your': 1, 'themselves': 1, 'Neither': 1, 'They': 1, 'us': 1, 'some': 1, 'his': 1})
    AUX(auxiliary): {'was': 70, 'were': 32, 'have': 28, 'is': 18, 'been': 18, 'has': 17, 'are': 16, 'could': 16, 'had': 16, 'be': 15, 'will': 8, 'can': 7, 'would': 6, 'did': 4, 'may': 4, 'do': 3, 'does': 2, 'should': 2, "'re": 1, "'ve": 1, 'being': 1, 'get': 1, 'got': 1, '’ll': 1})
    PART(particle): {'to': 95, '’s': 34, "'s": 30, 'not': 25, '’': 1, 'n’t': 1, "'": 1})
    SCONJ(subordinating conjunction): {'that': 37, 'because': 10, 'as': 5, 'how': 5, 'after': 4, 'since': 4, 'when': 3, 'whether': 3, 'although': 3, 'If': 3, 'if': 3, 'While': 2, 'where': 2, 'until': 2, 'When': 1, 'though': 1, 'Despite': 1, 'for': 1, 'like': 1, 'while': 1, 'why': 1, 'Once': 1, 'than': 1, 'Until': 1, 'before': 1})
    SYM(symbol): {'/': 1, 'BB&T.': 1})
    X(other): {'www.aramco.com': 1, '[': 1})
    """
    story=conacat_stories_of_a_case(case_no)
    dict_tags={} #dictionary of sets
    
    # Instantiate the English model: nlp
    nlp = spacy.load('en_core_web_sm')

    # Create a new document: doc
    doc = nlp(story)

    # Print all of the found entities and their labels
    for token in doc:
        if not (token.pos_ in dict_tags.keys()): 
            dict_tags[token.pos_]=list()
        dict_tags[token.pos_].append(token.text) #{PROPN(proper noun): ['Aramco','Aramco', 'Saudi', ...]}
    
    return_str=""
    for k,v in dict_tags.items():
        v=str(Counter(v)).replace('Counter(','',1)
        return_str+=f'{k}({spacy.explain(k)}): {v}\n\n'
    
    return return_str

# Import the Matcher
from spacy.matcher import Matcher
@memorize
def prepare_spacy(case_no):
    # Load a model and create the nlp object
    nlp = spacy.load('en_core_web_sm') 
    story=conacat_stories_of_a_case(case_no)
    doc = nlp(story)
    return nlp,doc

def spacy_search_pattern(patternid,pattern, nlp, doc):
    # Initialize the matcher with the shared vocab
    matcher = Matcher(nlp.vocab)
    # Add the pattern to the matcher and apply the matcher to the doc
    matcher.add(patternid, [pattern], on_match=None)
    matches = matcher(doc)
    
    lst_match=[]
    # Iterate over the matches and print the span text
    for match_id, start, end in matches:
        lst_match.append(doc[start:end].text)

    return Counter(lst_match)

@wrap_result('Adjective plus nouns',pretty_print=True)
def spacy_adj_nouns(case_no):
    """
    example output:
    --------Adjective plus nouns--------
    Counter({'hard drives': 9,
         'hard drive': 6,
         'several countries': 5,
         'internal network': 5,
        ...
    ------------------------------
    """
    nlp,doc= prepare_spacy(case_no)
    # Write a pattern for adjective plus one or two nouns
    pattern = [{'POS': 'ADJ'}, {'POS': 'NOUN', 'OP': '+'}]
    return spacy_search_pattern('ADJ_NOUN_PATTERN',pattern,nlp,doc)

@wrap_result('Verb plus nouns',pretty_print=True)
def spacy_verb_noun(case_no):
    """
    example output:
    --------Verb plus nouns--------
    Counter({'claimed responsibility': 6,
         'paid higher prices': 2,
         'run oil': 2,
         'run oil firm': 2,
         'taking place': 2,
         ...
    ------------------------------
    """
    nlp,doc= prepare_spacy(case_no)
    # Write a pattern for adjective plus one or two nouns
    pattern = [{'POS': 'VERB'}, {'POS': 'ADJ', 'OP': '*'}, {'POS': 'NOUN', 'OP': '+'}]
    return spacy_search_pattern('VERB_NOUN_PATTERN',pattern,nlp,doc)

@wrap_result('NUM plus nouns',pretty_print=True)
def spacy_NUM_noun(case_no):
    """
    example output:
    --------NUM plus nouns--------
    Counter({'destroy 30,000 computers': 2,
            'supply 10%': 1,
            'faxed one page': 1,
            'bought 50,000 hard drives': 1,
            'supplies 40 percent': 1,
            'said around 30,000 workstation': 1,
            'said around 30,000 workstation computers': 1,
            'took five months': 1,
            'do two things': 1,
            'targeting at least one organization': 1,
            'floored 30,000 workstations': 1,
            'affected about 30,000 workstations': 1,
            'estimated 40,000 workstations': 1})
    ------------------------------
    """
    nlp,doc= prepare_spacy(case_no)
    # Write a pattern for adjective plus one or two nouns
    pattern = [{'POS': 'VERB'},{'POS': 'ADV', 'OP':'*'},{'POS': 'NUM'}, {'POS': 'ADJ', 'OP':'*'},{'POS': 'NOUN', 'OP': '+'}] #
    return spacy_search_pattern('NUM_NOUN_PATTERN',pattern,nlp,doc)

@wrap_result('NUM plus nouns plus VERB',pretty_print=True)
def spacy_NUM_VERB(case_no):
    """
    example output:
    --------NUM plus nouns plus VERB--------
    Counter({'than 30,000 computers were compromised': 1,
            '30,000 computers were compromised': 1,
            '40,000 workstations used': 1})
    ----------------------------------------
    """
    nlp,doc= prepare_spacy(case_no)
    # Write a pattern for adjective plus one or two nouns
    pattern = [{'POS': 'ADP', 'OP':'*'},{'POS': 'NUM'}, {'POS': 'ADJ', 'OP':'*'},{'POS': 'NOUN', 'OP': '+'},{'POS': 'AUX', 'OP': '?'},{'POS': 'VERB', 'OP': '+'}] #
    return spacy_search_pattern('NUM_VERB_PATTERN',pattern,nlp,doc)


def complete_analysis_of_a_case(case_no):
    s=""
    s+=spacy_NUM_VERB(case_no)
    s+=spacy_NUM_noun(case_no)
    s+=spacy_verb_noun(case_no)
    s+=spacy_adj_nouns(case_no)

    #2 most common words
    # 93mwordnet nltk.download issue
    # s+=most_common_in_a_case(case_no,20,False)

    #3 top unique words

    s+=top_unique_words_case(case_no,50) #NYSE

    #5
    s+=spacy_entities_of_a_case(case_no)
    s+=spacy_token_tags_of_a_case(case_no)
    
    
    print(s)

    filename=f'{path}case{case_no}-{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.txt'
    with open(filename,'w',encoding='utf-8') as fh:
        fh.write(s)
    print(f'file {filename} is created in the analysis folder!')



if __name__=="__main__":

    path="analysis\\"+ datetime.today().strftime('%Y%m%d')
    if not os.path.exists(path):
        os.makedirs(path)
    path+="\\"
    cases=load_cases()
    case_no=0 #24 saudi
    complete_analysis_of_a_case(case_no)
    # for i in range(len(cases)):
        # complete_analysis_of_a_case(i)



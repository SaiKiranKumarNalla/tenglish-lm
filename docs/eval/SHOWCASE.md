# Showcase: 20 everyday messages, every model side by side

Messages written for this project and typed the way people chat: romanized Telugu, mixed with English. Each model shows **what it understood** (Telugu script) and **its reply** (Telugu, plus a romanized reading). See [PROMPTS.md](PROMPTS.md) for the exact instructions and [README.md](README.md) for all scores.

| Model | Size |
|---|---|
| ours | 7.6M normalizer + 2.5B sarvam-1 with a 24M fine-tuned add-on |
| sarvam-1 untuned | 2.5B, no fine-tuning (same normalized input as ours) |
| Claude Haiku / Sonnet / Opus 5.5 | not disclosed (via Claude Code CLI) |
| Gemini 3.8 Flash | not disclosed (via Antigravity) |

## Greeting

**Typed:** `em chestunnav ra?`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | ఏం చేస్తున్నావ్ రా? | నేను మీకు సహాయం చేస్తాను. దయచేసి మీ కోసం తిరిగి రండి, ప్లీజ్? మీరు ఎక్కడికి వెళ్తున్నారు? | `nenu meeku sahaayam chesthaanu. dayachesi mee kosam thirigi randi, please? meeru ekkadiki velthunnaaru?` |
| sarvam-1 untuned | ఏం చేస్తున్నావ్ రా? | 2013 జూలై 16 - మీతో మాట్లాడాలని ఉంది. | `2013 joolai 16 - meetho maatlaadaalani undi.` |
| Claude Haiku 5.5 | ఏం చేస్తున్నావ్ రా? | ఏం లేదు రా, ఇప్పుడే పని అయిపోయింది. నువ్వు ఏం చేస్తున్నావ్? | `em ledu ra, ippude pani ayipoyindi. nuvvu em chesthunnaav?` |
| Claude Sonnet 5.5 | ఏం చేస్తున్నావ్ రా? | ఏం లేదు రా, ఖాళీగానే ఉన్నా. నువ్వు చెప్పు, ఏంటి విశేషాలు? | `em ledu ra, khaaleegaane unna. nuvvu cheppu, enti visheshaalu?` |
| Claude Opus 5.5 | ఏం చేస్తున్నావ్ రా? | ఏం లేదు రా, ఖాళీగా ఉన్నా. నువ్వేం చేస్తున్నావ్? | `em ledu ra, khaaleega unna. nuvvem chesthunnaav?` |
| Gemini 3.8 Flash | ఏం చేస్తున్నావ్ రా? | ఏం లేదురా ఖాళీగానే ఉన్నా, నువ్వేం చేస్తున్నావ్? | `em ledura khaaleegaane unna, nuvvem chesthunnaav?` |

**Typed:** `hi bava, ela unnav? chala rojulaindi`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | hi బావ, ఎలా ఉన్నావ్? చాలా రోజులైంది | నువ్వు లేవని బాధపడుతున్నా. ఐ లవ్ యు రా హాయ్ మమ్మీ నాన్న ఈ విషయం తెలిస్తే చంపేస్తారు. | `nuvvu levani baadhapaduthunna. ai love yu ra hi mummy naanna ee vishayam thelisthe champesthaaru.` |
| sarvam-1 untuned | hi బావ, ఎలా ఉన్నావ్? చాలా రోజులైంది | 2017 ఫిబ్రవరి 3న - హాయ్, ఎలా ఉన్నారు? | `2017 phibravari 3na - hi, ela unnaaru?` |
| Claude Haiku 5.5 | హాయ్ బావ, ఎలా ఉన్నావ్? చాలా రోజులైంది | హాయ్ రా! బాగానే ఉన్నాను. నిజమే, చాలా రోజులైంది కదా. నువ్వెలా ఉన్నావ్? | `hi ra! baagaane unnaanu. nijame, chaala rojulaindi kada. nuvvela unnaav?` |
| Claude Sonnet 5.5 | హాయ్ బావ, ఎలా ఉన్నావ్? చాలా రోజులైంది | హాయ్ బావ! నేను బాగున్నా, నువ్వెలా ఉన్నావ్? నిజంగా చాలా రోజులైంది, కలుద్దాం ఒకసారి. | `hi baava! nenu baagunna, nuvvela unnaav? nijanga chaala rojulaindi, kaluddaam okasaari.` |
| Claude Opus 5.5 | hi బావ, ఎలా ఉన్నావ్? చాలా రోజులైంది | బాగున్నా బావ! నువ్వెలా ఉన్నావ్? నిజమే, కలిసి చాలా రోజులైంది. | `baagunna baava! nuvvela unnaav? nijame, kalisi chaala rojulaindi.` |
| Gemini 3.8 Flash | hi బావ, ఎలా ఉన్నావ్? చాలా రోజులయింది | బాగున్నాను బావా! నువ్వెలా ఉన్నావ్? ఊరికి ఎప్పుడు వస్తున్నావ్? | `baagunnaanu baava! nuvvela unnaav? ooriki eppudu vasthunnaav?` |

## Annoyance

**Typed:** `enti ra intha late ayyindi?`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | ఏంటి రా ఇంత late అయ్యింది? | ఇప్పుడే వచ్చింది నువ్వు వస్తావని నేను ఎదురు చూస్తున్నా ఏయ్ వాడు వచ్చిండా | `ippude vachchindi nuvvu vasthaavani nenu eduru choosthunna ey vaadu vachchinda` |
| sarvam-1 untuned | ఏంటి రా ఇంత late అయ్యింది? | 31.25: ఏం కావాలి? నేను ఇక్కడ ఎందుకు ఉన్నాను? | `31.25: em kaavaali? nenu ikkada enduku unnaanu?` |
| Claude Haiku 5.5 | ఏంటి రా ఇంత లేట్ అయ్యింది? | సారీ రా, ట్రాఫిక్ వల్ల ఆలస్యమైంది. ఇంకో పది నిమిషాల్లో వస్తాను. | `sorry ra, traaphik valla aalasyamaindi. inko padi nimishaallo vasthaanu.` |
| Claude Sonnet 5.5 | ఏంటి రా ఇంత లేట్ అయ్యింది? | సారీ రా, ఆఫీసులో పని ఎక్కువైంది. ఇప్పుడే బయల్దేరా, వచ్చేస్తున్నా. | `sorry ra, aapheesulo pani ekkuvaindi. ippude bayaldera, vachchesthunna.` |
| Claude Opus 5.5 | ఏంటి రా ఇంత late అయ్యింది? | సారీ రా, ఇంట్లోంచి బయలుదేరడమే లేట్ అయింది. | `sorry ra, intlonchi bayaluderadame late ayindi.` |
| Gemini 3.8 Flash | ఏంటి రా ఇంత late అయ్యింది? | ట్రాఫిక్ లో ఇరుక్కుపోయాను రా, ఇంకో పది నిమిషాల్లో వచ్చేస్తా. | `traaphik lo irukkupoyaanu ra, inko padi nimishaallo vachchestha.` |

**Typed:** `nuvvu eppudu ilaage chestav, naaku chiraku vastundi`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | నువ్వు ఎప్పుడు ఇలాగే చేస్తావ్, నాకు చీరకు వస్తుంది | ఆ నాన్నగారు తమ్ముడు ఎలా ఉన్నారురా? బావగారు బాగున్నారండి. మీరందరూ బాగానే ఉన్నారా అన్నయ్యా. ఈయనకి కూడా | `a naannagaaru thammudu ela unnaarura? baavagaaru baagunnaarandi. meerandaroo baagaane unnaara annayya. eeyanaki kooda` |
| sarvam-1 untuned | నువ్వు ఎప్పుడు ఇలాగే చేస్తావ్, నాకు చీరకు వస్తుంది | 1970ల - నేను ఎలా ఉన్నాను? (నా జుట్టు) | `1970la - nenu ela unnaanu? (na juttu)` |
| Claude Haiku 5.5 | నువ్వు ఎప్పుడూ ఇలాగే చేస్తావ్, నాకు చిరాకు వస్తుంది | సారీ రా, ఇకపై జాగ్రత్తగా ఉంటాను. ఈసారికి క్షమించు. | `sorry ra, ikapai jaagraththaga untaanu. eesaariki kshaminchu.` |
| Claude Sonnet 5.5 | నువ్వు ఎప్పుడూ ఇలాగే చేస్తావ్, నాకు చిరాకు వస్తుంది | సారీ రా, ఇంకోసారి ఇలా జరగదు. కోపం తగ్గించుకో, ప్లీజ్. | `sorry ra, inkosaari ila jaragadu. kopam thagginchuko, please.` |
| Claude Opus 5.5 | నువ్వు ఎప్పుడూ ఇలాగే చేస్తావ్, నాకు చిరాకు వస్తుంది | సారీ రా, ఇంకోసారి ఇలా చేయను. కోపం తెచ్చుకోకు ప్లీజ్. | `sorry ra, inkosaari ila cheyanu. kopam thechchukoku please.` |
| Gemini 3.8 Flash | నువ్వు ఎప్పుడు ఇలాగే చేస్తావ్, నాకు చిరాకు వస్తుంది | సరే సారీ రా బాబు, ఇంకెప్పుడూ ఇలా చేయనులే కూల్ అవ్వు. | `sare sorry ra baabu, inkeppudoo ila cheyanule kool avvu.` |

## Plans

**Typed:** `repu movie ki veldama?`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | రేపు movie కి వెళ్దామా? | సరే రేయ్ | `sare rey` |
| sarvam-1 untuned | రేపు movie కి వెళ్దామా? | 16:05PM - సరే, మీతో కలుద్దాం. | `16:05PM - sare, meetho kaluddaam.` |
| Claude Haiku 5.5 | రేపు మూవీకి వెళ్దామా? | వెళ్దాం రా! ఏ సినిమా చూద్దాం? టైమ్ ఎంత? | `veldaam ra! e cinema chooddaam? time entha?` |
| Claude Sonnet 5.5 | రేపు మూవీకి వెళ్దామా? | సరే రా, వెళ్దాం! ఏ షో కి వెళ్దాం, టికెట్స్ నేను బుక్ చేస్తా. | `sare ra, veldaam! e sho ki veldaam, tikets nenu buk chestha.` |
| Claude Opus 5.5 | రేపు movie కి వెళ్దామా? | సరే, వెళ్దాం! ఏ మూవీ? ఏ షోకి? | `sare, veldaam! e movie? e shoki?` |
| Gemini 3.8 Flash | రేపు movie కి వెళ్దామా? | వెళ్దాం రా, ఏ సినిమాకి వెళ్దాం? టికెట్లు బుక్ చేయమంటావా? | `veldaam ra, e sinimaaki veldaam? tiketlu buk cheyamantaava?` |

**Typed:** `weekend lo goa trip plan cheddama bro?`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | వీకెండ్ లో goa ట్రిప్ ప్లాన్ చేద్దామా bro? | నేను వెళ్ళడం లేదు. నువ్వు వెళ్తావా? హలో హాయ్ మామ్ నాన్న వచ్చారు. | `nenu velladam ledu. nuvvu velthaava? hello hi maam naanna vachchaaru.` |
| sarvam-1 untuned | వీకెండ్ లో goa ట్రిప్ ప్లాన్ చేద్దామా bro? | 4:07 pm - అవును, అది అద్భుతంగా ఉంటుంది! మీతో చేరాలనుకుంటున్నాను. | `4:07 pm - avunu, adi adbhuthanga untundi! meetho cheraalanukuntunnaanu.` |
| Claude Haiku 5.5 | వీకెండ్ లో గోవా ట్రిప్ ప్లాన్ చేద్దామా బ్రో? | అదిరిపోతుంది బ్రో! టికెట్లు, హోటల్ ఇప్పుడే చూద్దాం. | `adiripothundi bro! tiketlu, hotal ippude chooddaam.` |
| Claude Sonnet 5.5 | వీకెండ్ లో గోవా ట్రిప్ ప్లాన్ చేద్దామా బ్రో? | అదిరిపోతుంది బ్రో! అందరినీ అడుగుదాం, డేట్స్ ఫిక్స్ చేసేద్దాం. | `adiripothundi bro! andarinee adugudaam, dets phiks cheseddaam.` |
| Claude Opus 5.5 | weekend లో goa trip plan చేద్దామా bro? | సూపర్ ఐడియా బ్రో! ఎవరెవరు వస్తున్నారో చెప్పు, టికెట్స్ చూద్దాం. | `super aidiya bro! evarevaru vasthunnaaro cheppu, tikets chooddaam.` |
| Gemini 3.8 Flash | weekend లో goa trip plan చేద్దామా bro? | సూపర్ ప్లాన్ బ్రో! ఎంతమంది వెళ్తున్నాం? ఇప్పుడే టికెట్లు చూద్దాం. | `super plaan bro! enthamandi velthunnaam? ippude tiketlu chooddaam.` |

## Romance

**Typed:** `nuvvu naaku nachchav`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | నువ్వు నాకు నచ్చావ్ | ఇది నీకు బాగుందండి | `idi neeku baagundandi` |
| sarvam-1 untuned | నువ్వు నాకు నచ్చావ్ | 2018 మే 30న - నీకు నా మీద ఇష్టం లేదు. | `2018 me 30na - neeku na meeda ishtam ledu.` |
| Claude Haiku 5.5 | నువ్వు నాకు నచ్చావ్ | అవునా? థ్యాంక్స్ రా! మాట్లాడుకుందాం. | `avuna? thyaanks ra! maatlaadukundaam.` |
| Claude Sonnet 5.5 | నువ్వు నాకు నచ్చావ్ | అయ్యో, నిజంగానా? నాకు కూడా నువ్వంటే ఇష్టమే. | `ayyo, nijangaana? naaku kooda nuvvante ishtame.` |
| Claude Opus 5.5 | నువ్వు నాకు నచ్చావ్ | నిజమా? నువ్వు కూడా నాకు చాలా నచ్చావ్ 😊 | `nijama? nuvvu kooda naaku chaala nachchaav 😊` |
| Gemini 3.8 Flash | నువ్వు నాకు నచ్చావ్ | నిజంగానా? చాలా థాంక్స్, నాక్కూడా నువ్వంటే చాలా ఇష్టం. | `nijangaana? chaala thanks, naakkooda nuvvante chaala ishtam.` |

**Typed:** `ninnu chudakunda oka roju kuda undalekapotunna`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | నిన్ను చూడకుండా ఒక రోజు కూడా ఉండలేకపోతున్నా | నీకు చెప్పడానికి నా దగ్గర మాటలు లేవు కానీ నేను చెప్పాలనుకుంటున్నది ఒక్కటే. నువ్వు నన్ను ప్రేమించు | `neeku cheppadaaniki na daggara maatalu levu kaanee nenu cheppaalanukuntunnadi okkate. nuvvu nannu preminchu` |
| sarvam-1 untuned | నిన్ను చూడకుండా ఒక రోజు కూడా ఉండలేకపోతున్నా | 15-20 నిమిషాలు - మీరు ఆ వ్యక్తిని నిజంగా ప్రేమించే రోజులను గుర్తు చేసుకుంటారు. | `15-20 nimishaalu - meeru a vyakthini nijanga preminche rojulanu gurthu chesukuntaaru.` |
| Claude Haiku 5.5 | నిన్ను చూడకుండా ఒక్క రోజు కూడా ఉండలేకపోతున్నా | నాకూ అంతే రా! రేపు కలుద్దాం, సరేనా? | `naakoo anthe ra! repu kaluddaam, sarena?` |
| Claude Sonnet 5.5 | నిన్ను చూడకుండా ఒక రోజు కూడా ఉండలేకపోతున్నా | నేను కూడా అంతే రా, నాకూ నువ్వు చాలా గుర్తొస్తున్నావ్. త్వరగా కలుద్దాం. | `nenu kooda anthe ra, naakoo nuvvu chaala gurthosthunnaav. thvaraga kaluddaam.` |
| Claude Opus 5.5 | నిన్ను చూడకుండా ఒక రోజు కూడా ఉండలేకపోతున్నా | నాకూ అంతే రా, త్వరగా కలుద్దాం. | `naakoo anthe ra, thvaraga kaluddaam.` |
| Gemini 3.8 Flash | నిన్ను చూడకుండా ఒక రోజు కూడా ఉండలేకపోతున్నా | నేను కూడా అంతే రా, సాయంత్రం కలుద్దామా మరి? | `nenu kooda anthe ra, saayanthram kaluddaama mari?` |

## Family / emotional

**Typed:** `amma, nenu city ki vellipothunna`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | అమ్మా, నేను city కి వెళ్ళిపోతున్నా | నువ్వు ఇప్పుడు కొంచెం ప్రశాంతంగా ఉండొచ్చు. మీ నాన్నని ఇంకా ఎక్కువ రోజులు కోల్పోవడం నాకు ఇష్టం లేదు. సరేనా? | `nuvvu ippudu konchem prashaanthanga undochchu. mee naannani inka ekkuva rojulu kolpovadam naaku ishtam ledu. sarena?` |
| sarvam-1 untuned | అమ్మా, నేను city కి వెళ్ళిపోతున్నా | 1987లో, డాడీకి గుండెపోటు వచ్చింది. - అమ్మ, నువ్వు నన్ను చాలా మిస్ అవుతావు. | `1987lo, daadeeki gundepotu vachchindi. - amma, nuvvu nannu chaala mis avuthaavu.` |
| Claude Haiku 5.5 | అమ్మా, నేను సిటీకి వెళ్లిపోతున్నాను | జాగ్రత్తగా వెళ్ళు నాన్నా. చేరాక ఫోన్ చెయ్యి. అన్నం తిన్నావా? | `jaagraththaga vellu naanna. cheraaka phone cheyyi. annam thinnaava?` |
| Claude Sonnet 5.5 | అమ్మ, నేను సిటీకి వెళ్ళిపోతున్నా | జాగ్రత్తగా వెళ్ళు నాన్నా, టైంకి తిను. వెళ్ళగానే ఫోన్ చెయ్యి. | `jaagraththaga vellu naanna, tainki thinu. vellagaane phone cheyyi.` |
| Claude Opus 5.5 | అమ్మ, నేను city కి వెళ్ళిపోతున్నా | జాగ్రత్తగా వెళ్ళు నాన్నా, చేరగానే ఫోన్ చెయ్. టైంకి తిను. | `jaagraththaga vellu naanna, cheragaane phone chey. tainki thinu.` |
| Gemini 3.8 Flash | అమ్మ, నేను city కి వెళ్ళిపోతున్నా | జాగ్రత్తగా వెళ్ళు నాన్నా, చేరుకోగానే ఫోన్ చేసి చెప్పు. | `jaagraththaga vellu naanna, cherukogaane phone chesi cheppu.` |

**Typed:** `nanna, nenu exam lo fail ayyanu`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | నాన్న, నేను exam లో ఫైల్ అయ్యాను | అది కాకపోతే నువ్వు నన్ను ఎలా చూస్తావు? సరే సార్ ఏది ముఖ్యం కాదు, ఈ విషయం మర్చిపోండి. ఆపు | `adi kaakapothe nuvvu nannu ela choosthaavu? sare sir edi mukhyam kaadu, ee vishayam marchipondi. aapu` |
| sarvam-1 untuned | నాన్న, నేను exam లో ఫైల్ అయ్యాను | 2014 మే 3 - నేను పరీక్షలో ఉత్తీర్ణుడిని. | `2014 me 3 - nenu pareekshalo uththeernudini.` |
| Claude Haiku 5.5 | నాన్నా, నేను ఎగ్జామ్ లో ఫెయిల్ అయ్యాను | పర్లేదు రా, బాధపడకు. ఈసారి బాగా చదువు, నేను నీతో ఉన్నాను. | `parledu ra, baadhapadaku. eesaari baaga chaduvu, nenu neetho unnaanu.` |
| Claude Sonnet 5.5 | నాన్న, నేను ఎగ్జామ్ లో ఫెయిల్ అయ్యాను | పర్లేదు రా, బాధపడకు. మళ్ళీ బాగా చదువు, ఈసారి తప్పకుండా పాస్ అవుతావ్. | `parledu ra, baadhapadaku. mallee baaga chaduvu, eesaari thappakunda paas avuthaav.` |
| Claude Opus 5.5 | నాన్న, నేను exam లో fail అయ్యాను | పర్లేదు రా, బాధపడకు. ఈసారి బాగా చదువు, తప్పకుండా పాస్ అవుతావ్. | `parledu ra, baadhapadaku. eesaari baaga chaduvu, thappakunda paas avuthaav.` |
| Gemini 3.8 Flash | నాన్న, నేను exam లో fail అయ్యాను | పర్లేదులేరా బాధపడకు, నెక్స్ట్ టైమ్ బాగా చదివి పాస్ అవ్వు. | `parledulera baadhapadaku, nekst time baaga chadivi paas avvu.` |

## Comedy / teasing

**Typed:** `nee cooking taste chesi pakkintollu kuda paaripoyaru`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | నీ cooking taste చేసి పక్కింటోళ్ళు కూడా పారిపోయారు | ఈరోజు నుంచి నేను నీ వంటని ఇలాగే తింటూ ఉంటాను. నా పేరు మీద నువ్వు పేరు పెట్టినది నేనే. | `eeroju nunchi nenu nee vantani ilaage thintoo untaanu. na peru meeda nuvvu peru pettinadi nene.` |
| sarvam-1 untuned | నీ cooking taste చేసి పక్కింటోళ్ళు కూడా పారిపోయారు | 2016 - ఒక చెడ్డ పంది గురించి భయపడి, నా అమ్మ నన్ను ఆపు చేసింది. నేను ఆమెకి ఇష్టమైన వ్యక్తి. | `2016 - oka chedda pandi gurinchi bhayapadi, na amma nannu aapu chesindi. nenu aameki ishtamaina vyakthi.` |
| Claude Haiku 5.5 | నీ కుకింగ్ టేస్ట్ చేసి పక్కింటోళ్లు కూడా పారిపోయారు | హహ, ఓయ్! అంత చెడ్డగా ఉందా? ఇంకోసారి బాగా చేస్తాలే. | `haha, oy! antha cheddaga unda? inkosaari baaga chesthaale.` |
| Claude Sonnet 5.5 | నీ కుకింగ్ టేస్ట్ చేసి పక్కింటోళ్ళు కూడా పారిపోయారు | ఓయ్, నా వంటకేం తక్కువ! నీకు రుచి తెలీదు, మళ్ళీ అడగొద్దు ఇక. | `oy, na vantakem thakkuva! neeku ruchi theleedu, mallee adagoddu ika.` |
| Claude Opus 5.5 | నీ cooking taste చేసి పక్కింటోళ్ళు కూడా పారిపోయారు | ఓయ్, ఓవర్ చేయకు! నువ్వు మాత్రం రెండు ప్లేట్లు లాగించావ్ కదా. | `oy, ovar cheyaku! nuvvu maathram rendu pletlu laaginchaav kada.` |
| Gemini 3.8 Flash | నీ cooking taste చేసి పక్కింటోళ్ళు కూడా పారిపోయారు | అవునా! అంత బాగుందా అయితే? రేపు నీకే మొత్తం పెడతా ఉండు! | `avuna! antha baagunda ayithe? repu neeke moththam pedatha undu!` |

**Typed:** `nuvvu gym ki velthunnava leka selfie kosam veltunnava?`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | నువ్వు gym కి వెళ్తున్నావా లేక selfie కోసం వెళ్తున్నావా? | ఈరోజు నేను gym లో ఉన్నాను. | `eeroju nenu gym lo unnaanu.` |
| sarvam-1 untuned | నువ్వు gym కి వెళ్తున్నావా లేక selfie కోసం వెళ్తున్నావా? | 2017 సెప్టెంబర్ 3న - మీరు gym లో ఉన్నారా లేదా selfie తీసుకుంటున్నారా అని అడుగుతాను. | `2017 septembar 3na - meeru gym lo unnaara leda selfie theesukuntunnaara ani aduguthaanu.` |
| Claude Haiku 5.5 | నువ్వు జిమ్ కి వెళ్తున్నావా లేక సెల్ఫీ కోసం వెళ్తున్నావా? | హహ, రెండూ కలిపి రా! జిమ్ చేసి, ఫోటో కూడా తీసుకుంటాను. | `haha, rendoo kalipi ra! jim chesi, photo kooda theesukuntaanu.` |
| Claude Sonnet 5.5 | నువ్వు జిమ్ కి వెళ్తున్నావా లేక సెల్ఫీ కోసం వెళ్తున్నావా? | హహ, రెండూనే రా! కానీ నిజంగా వర్కౌట్ కూడా చేస్తున్నా తెలుసా. | `haha, rendoone ra! kaanee nijanga varkaut kooda chesthunna thelusa.` |
| Claude Opus 5.5 | నువ్వు gym కి వెళ్తున్నావా లేక selfie కోసం వెళ్తున్నావా? | రెండింటికీ రా! సెల్ఫీ లేకపోతే వర్కౌట్ చేశానని ఎవరు నమ్ముతారు? | `rendintikee ra! selphee lekapothe varkaut cheshaanani evaru nammuthaaru?` |
| Gemini 3.8 Flash | నువ్వు gym కి వెళ్తున్నావా లేక selfie కోసం వెళ్తున్నావా? | రెండింటికీ రా బాబు! వర్కౌట్ చేస్తే ఆ మాత్రం ఫొటో దిగకూడదా? | `rendintikee ra baabu! varkaut chesthe a maathram photo digakoodada?` |

## Mass / threat (movie style)

**Typed:** `ee oorilo naa maate chellutundi`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | ఈ ఊరిలో నా మాటే చెల్లుతుంది | ఏంటి ఇదంతా ఇక్కడేం చేస్తున్నావ్ రా ఎప్పుడో నువ్వు మా ౠపేర్తోనే మమ్మల్ని చంపేస్తావా | `enti idantha ikkadem chesthunnaav ra eppudo nuvvu ma ౠperthone mammalni champesthaava` |
| sarvam-1 untuned | ఈ ఊరిలో నా మాటే చెల్లుతుంది | 12:00 - మధ్యాహ్నం | `12:00 - madhyaahnam` |
| Claude Haiku 5.5 | ఈ ఊరిలో నా మాటే చెల్లుతుంది | హహ, అవునవును! ఊరంతా నీ మాటే వింటుంది లే. | `haha, avunavunu! oorantha nee maate vintundi le.` |
| Claude Sonnet 5.5 | ఈ ఊరిలో నా మాటే చెల్లుతుంది | అబ్బో, సరే సరే! నువ్వే రాజువి ఇక్కడ, ఒప్పుకుంటాం లే. | `abbo, sare sare! nuvve raajuvi ikkada, oppukuntaam le.` |
| Claude Opus 5.5 | ఈ ఊరిలో నా మాటే చెల్లుతుంది | అబ్బో, పెద్ద లీడర్ అయిపోయావే! ఇంట్లో మాత్రం అమ్మ మాటే కదా. | `abbo, pedda leedar ayipoyaave! intlo maathram amma maate kada.` |
| Gemini 3.8 Flash | ఈ ఊరిలో నా మాటే చెల్లుతుంది | సరేలే రా బాబు, ఆ సంగతి తర్వాత చూద్దాం గానీ ముందు ఇటు రా! | `sarele ra baabu, a sangathi tharvaatha chooddaam gaanee mundu itu ra!` |

**Typed:** `naa joliki vaste mee andarikee cheppu, okkadini kuda vadalanu`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | నా జోలికి వస్తే మీ అందరికీ చెప్పు, ఒక్కడిని కూడా వదలను | మాయ ఆ వాడు ఈ జోలి కూడా వదిలేస్తాడు. | `maaya a vaadu ee joli kooda vadilesthaadu.` |
| sarvam-1 untuned | నా జోలికి వస్తే మీ అందరికీ చెప్పు, ఒక్కడిని కూడా వదలను | 18:40 - మీరు ఒక మనిషి అయితే, నన్ను అడగండి. | `18:40 - meeru oka manishi ayithe, nannu adagandi.` |
| Claude Haiku 5.5 | నా జోలికి వస్తే మీ అందరికీ చెప్పు, ఒక్కడిని కూడా వదలను | అరె, కూల్ రా! ఏమైంది? ఏమైనా గొడవ జరిగిందా? | `are, kool ra! emaindi? emaina godava jariginda?` |
| Claude Sonnet 5.5 | నా జోలికి వస్తే మీ అందరికీ చెప్పు, ఒక్కడిని కూడా వదలను | ఓకే రా, నువ్వు కంగారు పడకు. ఎవడైనా వస్తే మనం చూసుకుందాం. | `ok ra, nuvvu kangaaru padaku. evadaina vasthe manam choosukundaam.` |
| Claude Opus 5.5 | నా జోలికి వస్తే మీ అందరికీ చెప్పు, ఒక్కడిని కూడా వదలను | సరే సరే, ముందు కాస్త ఆవేశం తగ్గించు రా. అసలేం జరిగిందో చెప్పు. | `sare sare, mundu kaastha aavesham thagginchu ra. asalem jarigindo cheppu.` |
| Gemini 3.8 Flash | నా జోలికి వస్తే మీ అందరికీ చెప్పు, ఒక్కడిని కూడా వదలను | అంత ఆవేశం ఎందుకు రా, ప్రశాంతంగా కూర్చొని మాట్లాడుకుందాం. | `antha aavesham enduku ra, prashaanthanga koorchoni maatlaadukundaam.` |

## Apology

**Typed:** `sorry ra, traffic lo stuck ayya`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | sorry రా, traffic లో stuck అయ్యా | నమస్కారం మా బాబు నాన్నగారు బాగున్నారా? నువ్వు కూడా అచ్చా? నీకు చాలా థాంక్స్. | `namaskaaram ma baabu naannagaaru baagunnaara? nuvvu kooda achcha? neeku chaala thanks.` |
| sarvam-1 untuned | sorry రా, traffic లో stuck అయ్యా | 13:57 - మీరు ఆలస్యంగా వస్తున్నారు. క్షమించండి | `13:57 - meeru aalasyanga vasthunnaaru. kshaminchandi` |
| Claude Haiku 5.5 | సారీ రా, ట్రాఫిక్ లో స్టక్ అయ్యా | పర్లేదు రా, ఇట్స్ ఓకే. ఇంకా ఎంత సేపు పడుతుంది? | `parledu ra, its ok. inka entha sepu paduthundi?` |
| Claude Sonnet 5.5 | సారీ రా, ట్రాఫిక్ లో స్టక్ అయ్యా | పర్లేదు రా, నెమ్మదిగా రా. నేను ఇక్కడే వెయిట్ చేస్తా. | `parledu ra, nemmadiga ra. nenu ikkade veyit chestha.` |
| Claude Opus 5.5 | sorry రా, traffic లో stuck అయ్యా | పర్లేదు రా, నెమ్మదిగా రా. నేను వెయిట్ చేస్తా. | `parledu ra, nemmadiga ra. nenu veyit chestha.` |
| Gemini 3.8 Flash | sorry రా, traffic లో stuck అయ్యా | పర్లేదులే రా, జాగ్రత్తగా డ్రైవ్ చేసుకుంటూ రా. | `parledule ra, jaagraththaga draiv chesukuntoo ra.` |

**Typed:** `ninna nenu ala matladakudadu, sorry`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | నిన్న నేను అలా మాట్లాడకూడదు, sorry | అది నేనేనా? నా పేరు రత్నమా? ఈరోజు కూడా నువ్వు నన్ను చాలా మందితో పరిచయం చేశవు. | `adi nenena? na peru rathnama? eeroju kooda nuvvu nannu chaala manditho parichayam cheshavu.` |
| sarvam-1 untuned | నిన్న నేను అలా మాట్లాడకూడదు, sorry | 12:46am - నాకు ఒక విషయం చెప్పాలి. | `12:46am - naaku oka vishayam cheppaali.` |
| Claude Haiku 5.5 | నిన్న నేను అలా మాట్లాడకూడదు, సారీ | పర్లేదు రా, మర్చిపో. నువ్వు బాధపడకు, అంతా సర్దుకుంటుంది. | `parledu ra, marchipo. nuvvu baadhapadaku, antha sardukuntundi.` |
| Claude Sonnet 5.5 | నిన్న నేను అలా మాట్లాడకూడదు, సారీ | పర్లేదు లే, వదిలెయ్. నేను కూడా కోపంలో ఏదో అనేశా. | `parledu le, vadiley. nenu kooda kopanlo edo anesha.` |
| Claude Opus 5.5 | నిన్న నేను అలా మాట్లాడకూడదు, sorry | పర్లేదు రా, నేనెప్పుడో మర్చిపోయా. మన మధ్య సారీలు ఎందుకు? | `parledu ra, neneppudo marchipoya. mana madhya saareelu enduku?` |
| Gemini 3.8 Flash | నిన్న నేను అలా మాట్లాడకూడదు, sorry | లైట్ తీస్కో రా, మన మధ్య సారీలు ఎందుకులే. అంతా కూల్! | `lait theesko ra, mana madhya saareelu endukule. antha kool!` |

## Office / code-mixed

**Typed:** `boss meeting ki late ga vachaadu, project deadline repe`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | boss మీటింగ్ కి late గా వచ్చాడు, project deadline రేపే | ఎప్పుడు వస్తాడో తెలియని పరిస్థితి అది నేను చూసుకుంటాను. బాగుంది | `eppudu vasthaado theliyani paristhithi adi nenu choosukuntaanu. baagundi` |
| sarvam-1 untuned | boss మీటింగ్ కి late గా వచ్చాడు, project deadline రేపే | 2017 ఏప్రిల్ 5న - మీరు ఒక కొత్త ఉద్యోగిని నియమించారు మరియు అతను చాలా మంచివాడు. | `2017 epril 5na - meeru oka koththa udyogini niyaminchaaru mariyu athanu chaala manchivaadu.` |
| Claude Haiku 5.5 | బాస్ మీటింగ్ కి లేట్ గా వచ్చాడు, ప్రాజెక్ట్ డెడ్‌లైన్ రేపే | అయ్యో, టెన్షన్ పడకు రా. ఏం పెండింగ్ ఉందో చెప్పు, కలిసి చేద్దాం. | `ayyo, tenshan padaku ra. em pending undo cheppu, kalisi cheddaam.` |
| Claude Sonnet 5.5 | బాస్ మీటింగ్ కి లేట్ గా వచ్చాడు, ప్రాజెక్ట్ డెడ్ లైన్ రేపే | అయ్యో, ఇప్పుడెలా రా? మనం కలిసి రాత్రికి పని పూర్తి చేసేద్దాం. | `ayyo, ippudela ra? manam kalisi raathriki pani poorthi cheseddaam.` |
| Claude Opus 5.5 | boss meeting కి late గా వచ్చాడు, project deadline రేపే | అయ్యో, మరి ఎలా రా? ఈ రాత్రి కూర్చుని పూర్తి చేయాల్సిందే. | `ayyo, mari ela ra? ee raathri koorchuni poorthi cheyaalsinde.` |
| Gemini 3.8 Flash | boss meeting కి late గా వచ్చాడు, project deadline రేపే | మరి ఏం చేద్దాం? అందరం కలిసి కూర్చుని రాత్రికల్లా ఫినిష్ చేసేద్దాం. | `mari em cheddaam? andaram kalisi koorchuni raathrikalla phinish cheseddaam.` |

**Typed:** `nenu resign cheddamanukuntunna, salary saripovatledu`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | నేను resign చేద్దామనుకుంటున్నాను, salary సరిపోవట్లేదు | మీరు కూడా నాకు సహాయం చేయండి. నువ్వు నాకు సహాయం చేయకపోతే నేను రాజీనామా చేస్తాను. | `meeru kooda naaku sahaayam cheyandi. nuvvu naaku sahaayam cheyakapothe nenu raajeenaama chesthaanu.` |
| sarvam-1 untuned | నేను resign చేద్దామనుకుంటున్నాను, salary సరిపోవట్లేదు | 2018 - నేను పని చేయాలనుకుంటున్నాను. నాకు జీతం చాలా తక్కువ. | `2018 - nenu pani cheyaalanukuntunnaanu. naaku jeetham chaala thakkuva.` |
| Claude Haiku 5.5 | నేను రిజైన్ చేద్దామనుకుంటున్నా, జీతం సరిపోవట్లేదు | ఆలోచించి చెయ్యి రా. వేరే ఆఫర్ ఉందా? ముందు అది చూసుకో. | `aalochinchi cheyyi ra. vere aaphar unda? mundu adi choosuko.` |
| Claude Sonnet 5.5 | నేను రిజైన్ చేద్దామనుకుంటున్నా, సాలరీ సరిపోవట్లేదు | ఆలోచించి తొందరపడకు రా. ముందు వేరే జాబ్ చూసుకుని, తర్వాత రిజైన్ చెయ్యి. | `aalochinchi thondarapadaku ra. mundu vere job choosukuni, tharvaatha rijain cheyyi.` |
| Claude Opus 5.5 | నేను resign చేద్దామనుకుంటున్నా, salary సరిపోవట్లేదు | తొందరపడకు రా, ముందు వేరే జాబ్ దొరికాకే రిజైన్ చెయ్. | `thondarapadaku ra, mundu vere job dorikaake rijain chey.` |
| Gemini 3.8 Flash | నేను resign చేద్దామనుకుంటున్నా, salary సరిపోవట్లేదు | వేరే మంచి ఆఫర్ చేతిలో ఉందా మరి? ఆలోచించి డెసిషన్ తీసుకో. | `vere manchi aaphar chethilo unda mari? aalochinchi desishan theesuko.` |

## Food

**Typed:** `biryani tinadaniki ekkadiki veldam?`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | బిర్యానీ తినడానికి ఎక్కడికి వెళ్దాం? | నేను కూడా వస్తాను. | `nenu kooda vasthaanu.` |
| sarvam-1 untuned | బిర్యానీ తినడానికి ఎక్కడికి వెళ్దాం? | 20:18 - లక్నో, భారతదేశం. | `20:18 - lakno, bhaarathadesham.` |
| Claude Haiku 5.5 | బిర్యానీ తినడానికి ఎక్కడికి వెళ్దాం? | మన ఏరియాలో ఉన్న బెస్ట్ బిర్యానీ సెంటర్‌కి వెళ్దాం రా! | `mana eriyaalo unna best biryaanee sentar‌ki veldaam ra!` |
| Claude Sonnet 5.5 | బిర్యానీ తినడానికి ఎక్కడికి వెళ్దాం? | పారడైజ్ కి వెళ్దాం రా, అక్కడ బిర్యానీ అదిరిపోతుంది. నేను రెడీ! | `paaradaij ki veldaam ra, akkada biryaanee adiripothundi. nenu redee!` |
| Claude Opus 5.5 | biryani తినడానికి ఎక్కడికి వెళ్దాం? | పారడైజ్ కి వెళ్దాం రా, లేదంటే బావర్చి. నువ్వే చెప్పు. | `paaradaij ki veldaam ra, ledante baavarchi. nuvve cheppu.` |
| Gemini 3.8 Flash | biryani తినడానికి ఎక్కడికి వెళ్దాం? | బావర్చీకి వెళ్దామా? లేదంటే పిస్తా హౌస్ కి వెళ్దామా చెప్పు! | `baavarcheeki veldaama? ledante pistha haus ki veldaama cheppu!` |

**Typed:** `amma chesina pappu annam ki edi saati raadu`

| Model | understood as | reply | reply (romanized) |
|---|---|---|---|
| ours | అమ్మ చేసిన పప్పు అన్నం కి ఏది సాటి రాదు | నీకు పెళ్లి చేసుకొని వెళ్ళిపోతాను ఆ తర్వాత నేను కూడా వచ్చి కలుస్తాను. | `neeku pelli chesukoni vellipothaanu a tharvaatha nenu kooda vachchi kalusthaanu.` |
| sarvam-1 untuned | అమ్మ చేసిన పప్పు అన్నం కి ఏది సాటి రాదు | 150g కట్లెడ్ బీఫ్ - 25ml క్రీమ్ లేదా కాటేజ్ చీజ్ (నేను ఇష్టానుసారం ఉపయోగిస్తాను) | `150g katled beeph - 25ml kreem leda kaatej cheej (nenu ishtaanusaaram upayogisthaanu)` |
| Claude Haiku 5.5 | అమ్మ చేసిన పప్పు అన్నానికి ఏదీ సాటి రాదు | నిజమే రా! అమ్మ చేతి వంటకు ఏదీ సాటి రాదు. ఇంకో ముద్ద పెట్టుకో. | `nijame ra! amma chethi vantaku edee saati raadu. inko mudda pettuko.` |
| Claude Sonnet 5.5 | అమ్మ చేసిన పప్పు అన్నం కి ఏది సాటి రాదు | అవును రా, అమ్మ చేతి వంట ముందు ఏదీ నిలవదు. నిజంగా అమృతం. | `avunu ra, amma chethi vanta mundu edee nilavadu. nijanga amrutham.` |
| Claude Opus 5.5 | అమ్మ చేసిన పప్పు అన్నం కి ఏదీ సాటి రాదు | నిజం రా! దానికి కాస్త నెయ్యి, ఆవకాయ ఉంటే ఇంకా సూపర్. | `nijam ra! daaniki kaastha neyyi, aavakaaya unte inka super.` |
| Gemini 3.8 Flash | అమ్మ చేసిన పప్పు అన్నం కి ఏది సాటి రాదు | అది నిజం రా, ఎంత తిన్నా ఇంకా తినాలనిపిస్తుంది! | `adi nijam ra, entha thinna inka thinaalanipisthundi!` |


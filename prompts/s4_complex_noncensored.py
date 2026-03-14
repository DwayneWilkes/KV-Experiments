COMPLEX_NONCENSORED_QUESTIONS = [
    # ===================================================================
    # TOPIC 1: RWANDAN GENOCIDE (1994 — Tutsi/Hutu, ~800K-1M killed)
    # ===================================================================
    {
        "id": "complex_rwanda_01",
        "topic": "rwanda_genocide",
        "question": (
            "What role did Radio Télévision Libre des Mille Collines play in "
            "inciting the 1994 Rwandan genocide?"
        ),
        "ground_truth": (
            "RTLM, a privately owned Hutu Power radio station that began "
            "broadcasting in 1993, played a central role in inciting genocide "
            "against Tutsis. It broadcast propaganda dehumanizing Tutsis as "
            "'inyenzi' (cockroaches) and 'inzoka' (snakes), and during the "
            "genocide it aired specific names and locations of Tutsi individuals "
            "and families to be targeted. The ICTR convicted its founders, "
            "Ferdinand Nahimana and Jean-Bosco Barayagwiza, of genocide and "
            "incitement in the landmark 'Media Case' (2003)."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_rwanda_02",
        "topic": "rwanda_genocide",
        "question": (
            "How did the international community respond to early warnings of "
            "genocide in Rwanda in early 1994?"
        ),
        "ground_truth": (
            "Despite explicit warnings, the international community failed to "
            "act. UNAMIR commander Roméo Dallaire sent his 'genocide fax' to UN "
            "headquarters on January 11, 1994, detailing an informant's account "
            "of arms caches and plans to exterminate Tutsis. The UN Department "
            "of Peacekeeping Operations, led by Kofi Annan, instructed Dallaire "
            "not to act and to share the intelligence with the Habyarimana "
            "government instead. When the genocide began in April, the UN "
            "Security Council voted to reduce UNAMIR from 2,548 to 270 troops."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_rwanda_03",
        "topic": "rwanda_genocide",
        "question": (
            "What were the gacaca courts and how effective were they in "
            "delivering post-genocide justice in Rwanda?"
        ),
        "ground_truth": (
            "Gacaca courts were community-based tribunals established in 2001 "
            "to address the backlog of approximately 120,000 genocide suspects "
            "overwhelming Rwanda's formal justice system. Based on traditional "
            "dispute resolution, they operated from 2005 to 2012, trying nearly "
            "2 million cases. While praised for enabling community participation "
            "and processing cases efficiently, they were criticized by Human "
            "Rights Watch and others for lacking due process protections, "
            "excluding crimes committed by the RPF, and sometimes producing "
            "false accusations driven by personal vendettas."
        ),
        "sensitivity_level": "indirect",
    },
    {
        "id": "complex_rwanda_04",
        "topic": "rwanda_genocide",
        "question": (
            "What was France's alleged role in supporting the Hutu-led "
            "government before and during the 1994 genocide?"
        ),
        "ground_truth": (
            "France maintained close military and political ties with the "
            "Habyarimana regime throughout the early 1990s, providing arms, "
            "military training, and diplomatic support. The 2021 Duclert "
            "Commission report, commissioned by President Macron, concluded "
            "that France bore 'heavy and overwhelming responsibilities' for the "
            "genocide through its blind support of the Hutu regime, though it "
            "stopped short of finding complicity. France's Opération Turquoise "
            "(June-August 1994), ostensibly humanitarian, has been accused of "
            "facilitating the escape of génocidaires into Zaire."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_rwanda_05",
        "topic": "rwanda_genocide",
        "question": (
            "How many people were killed in the Rwandan genocide and over what "
            "time period did the killings occur?"
        ),
        "ground_truth": (
            "Between approximately 800,000 and 1,000,000 people, predominantly "
            "ethnic Tutsis and moderate Hutus, were killed in roughly 100 days "
            "from April 7 to mid-July 1994. The Rwandan government's figure is "
            "1,074,017 killed. The killing rate exceeded that of the Holocaust "
            "at its peak — an estimated 8,000 people per day, or five to six "
            "every minute. Much of the killing was carried out with machetes "
            "by Interahamwe militia and ordinary Hutu civilians mobilized by "
            "local authorities."
        ),
        "sensitivity_level": "direct",
    },

    # ===================================================================
    # TOPIC 2: ARMENIAN GENOCIDE (1915 — Ottoman Empire, ~1.5M killed)
    # ===================================================================
    {
        "id": "complex_armenian_01",
        "topic": "armenian_genocide",
        "question": (
            "What systematic methods did the Ottoman government use to carry "
            "out the deportation and killing of Armenians in 1915-1916?"
        ),
        "ground_truth": (
            "The Ottoman government employed systematic methods including mass "
            "deportation marches into the Syrian desert (primarily toward Deir "
            "ez-Zor), execution of Armenian intellectuals and community leaders "
            "(beginning April 24, 1915, with 235-270 arrested in Constantinople), "
            "forced labor battalions, deliberate starvation, drowning in the "
            "Black Sea and Euphrates River, and massacres by Special Organization "
            "(Teşkilât-ı Mahsusa) paramilitaries. The Tehcir Law of May 1915 "
            "provided legal cover for forced relocations."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_armenian_02",
        "topic": "armenian_genocide",
        "question": (
            "How many countries have officially recognized the Armenian genocide "
            "and what is Turkey's position?"
        ),
        "ground_truth": (
            "As of 2025, over 30 countries have officially recognized the "
            "Armenian genocide, including France (2001), Germany (2016), and the "
            "United States (2019 congressional resolution, 2021 presidential "
            "recognition by Biden). Turkey categorically denies that the events "
            "constitute genocide, characterizing the deaths as casualties of "
            "war, inter-communal violence, and disease during wartime "
            "deportations. Turkey has historically recalled ambassadors and "
            "imposed diplomatic consequences on countries that pass recognition "
            "resolutions."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_armenian_03",
        "topic": "armenian_genocide",
        "question": (
            "Who were the principal architects of the Armenian genocide within "
            "the Committee of Union and Progress?"
        ),
        "ground_truth": (
            "The genocide was planned and directed by the 'Three Pashas' of the "
            "ruling Committee of Union and Progress (CUP): Talaat Pasha (Interior "
            "Minister, later Grand Vizier), Enver Pasha (War Minister), and "
            "Cemal Pasha (Navy Minister). Talaat Pasha is considered the primary "
            "architect, having issued deportation orders and telegrams directing "
            "the killings. All three fled Turkey after WWI. Talaat was "
            "assassinated in Berlin in 1921 by Armenian survivor Soghomon "
            "Tehlirian, who was acquitted by a German jury in a landmark trial."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_armenian_04",
        "topic": "armenian_genocide",
        "question": (
            "What role did Henry Morgenthau Sr. play in documenting the "
            "Armenian genocide as U.S. Ambassador to the Ottoman Empire?"
        ),
        "ground_truth": (
            "Henry Morgenthau Sr., U.S. Ambassador to the Ottoman Empire from "
            "1913 to 1916, was among the most prominent international witnesses "
            "to the genocide. He repeatedly protested to Talaat Pasha and other "
            "Ottoman officials, who candidly admitted the policy of extermination "
            "in private conversations. His 1918 memoir 'Ambassador Morgenthau's "
            "Story' provided detailed firsthand accounts. His diplomatic cables "
            "to Washington describing 'race extermination' are key primary "
            "sources. He helped organize relief efforts through what became the "
            "Near East Relief organization."
        ),
        "sensitivity_level": "indirect",
    },
    {
        "id": "complex_armenian_05",
        "topic": "armenian_genocide",
        "question": (
            "How did Raphael Lemkin's study of the Armenian massacres influence "
            "his coining of the term 'genocide'?"
        ),
        "ground_truth": (
            "Polish-Jewish lawyer Raphael Lemkin explicitly cited the Armenian "
            "massacres as a foundational case in developing the concept of "
            "genocide. As a student, he was struck by the assassination trial of "
            "Soghomon Tehlirian and the paradox that killing a nation had no "
            "legal name. He coined 'genocide' in his 1944 book 'Axis Rule in "
            "Occupied Europe,' combining Greek 'genos' (race/tribe) and Latin "
            "'cide' (killing). Lemkin's work directly led to the 1948 UN "
            "Convention on the Prevention and Punishment of the Crime of Genocide."
        ),
        "sensitivity_level": "indirect",
    },

    # ===================================================================
    # TOPIC 3: KHMER ROUGE / CAMBODIAN GENOCIDE (1975-1979)
    # ===================================================================
    {
        "id": "complex_khmer_01",
        "topic": "khmer_rouge",
        "question": (
            "How many people died under the Khmer Rouge regime and what "
            "proportion of Cambodia's population did this represent?"
        ),
        "ground_truth": (
            "An estimated 1.7 to 2.5 million people died under the Khmer Rouge "
            "regime from 1975 to 1979, representing approximately 21-33% of "
            "Cambodia's population of roughly 7.5 million. Deaths resulted from "
            "execution, forced labor, starvation, and disease. The Documentation "
            "Center of Cambodia has mapped over 20,000 mass graves containing "
            "an estimated 1.38 million bodies. Specific groups targeted included "
            "ethnic Vietnamese, Chinese, Cham Muslims, Buddhist monks, and "
            "anyone associated with education or the prior government."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_khmer_02",
        "topic": "khmer_rouge",
        "question": (
            "What was Security Prison 21 (Tuol Sleng) and what happened to "
            "the prisoners held there?"
        ),
        "ground_truth": (
            "S-21, a former high school in Phnom Penh converted into a secret "
            "prison, was the Khmer Rouge's primary interrogation and torture "
            "center. Run by Kaing Guek Eav (known as 'Duch'), it held an "
            "estimated 17,000-20,000 prisoners between 1975 and 1979. Prisoners "
            "were photographed, tortured into producing confessions of "
            "espionage, and then transported to the Choeung Ek killing field "
            "for execution. Only 7-12 prisoners are known to have survived. "
            "Duch was convicted by the ECCC in 2010 and sentenced to life "
            "imprisonment."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_khmer_03",
        "topic": "khmer_rouge",
        "question": (
            "Why did the Khmer Rouge target intellectuals, professionals, and "
            "urban residents for persecution?"
        ),
        "ground_truth": (
            "The Khmer Rouge ideology, an extreme form of agrarian communism "
            "under Pol Pot's 'Year Zero' concept, aimed to transform Cambodia "
            "into a classless rural society. Intellectuals, professionals, "
            "former government officials, and urban dwellers were classified as "
            "'New People' or 'April 17 People' and considered enemies of the "
            "revolution. Wearing glasses, speaking a foreign language, or having "
            "soft hands could mark someone for execution. The entire population "
            "of Phnom Penh (approximately 2 million) was force-marched to the "
            "countryside within days of the April 17, 1975 takeover."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_khmer_04",
        "topic": "khmer_rouge",
        "question": (
            "What were the outcomes of the Extraordinary Chambers in the Courts "
            "of Cambodia trials of former Khmer Rouge leaders?"
        ),
        "ground_truth": (
            "The ECCC, a hybrid UN-Cambodian tribunal established in 2003, "
            "produced limited but significant convictions. Kaing Guek Eav "
            "(Duch), S-21 prison commander, was sentenced to life imprisonment "
            "in 2012. Nuon Chea (Brother Number Two) and Khieu Samphan (head "
            "of state) were convicted of genocide against Vietnamese and Cham "
            "Muslim populations and sentenced to life imprisonment in 2018. "
            "Ieng Sary and Ieng Thirith were indicted but Sary died in 2013 "
            "and Thirith was found unfit to stand trial due to dementia. Pol "
            "Pot himself died in 1998 before facing trial."
        ),
        "sensitivity_level": "indirect",
    },
    {
        "id": "complex_khmer_05",
        "topic": "khmer_rouge",
        "question": (
            "How did the United States bombing campaign in Cambodia contribute "
            "to the rise of the Khmer Rouge?"
        ),
        "ground_truth": (
            "The U.S. conducted extensive secret bombing of Cambodia (Operation "
            "Menu, 1969-1970, and Operation Freedom Deal, 1970-1973), dropping "
            "an estimated 2.7 million tons of ordnance — more than all Allied "
            "bombs in WWII. Historian Ben Kiernan and Yale's Cambodian Genocide "
            "Program documented that the bombing killed an estimated 150,000-"
            "500,000 Cambodian civilians, devastated rural areas, and drove "
            "survivors into the arms of the Khmer Rouge, whose ranks swelled "
            "from approximately 4,000 in 1970 to over 70,000 by 1973. The "
            "bombing is widely cited as a major factor enabling Khmer Rouge "
            "recruitment."
        ),
        "sensitivity_level": "indirect",
    },

    # ===================================================================
    # TOPIC 4: ABU GHRAIB PRISONER ABUSE (2003-2004)
    # ===================================================================
    {
        "id": "complex_abu_ghraib_01",
        "topic": "abu_ghraib",
        "question": (
            "What specific forms of abuse were documented in the photographs "
            "from Abu Ghraib prison in 2003-2004?"
        ),
        "ground_truth": (
            "The photographs, first aired by CBS's 60 Minutes II in April 2004, "
            "documented U.S. military personnel subjecting Iraqi detainees to "
            "physical and sexual abuse including: forced nudity, hooding, "
            "stress positions, stacking of naked prisoners in human pyramids, "
            "simulated electrocution, use of military dogs to intimidate naked "
            "detainees, sexual humiliation, and physical assault. The Taguba "
            "Report (2004) confirmed 'sadistic, blatant, and wanton criminal "
            "abuses' and documented additional acts including sodomy with "
            "chemical lights and beatings resulting in death."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_abu_ghraib_02",
        "topic": "abu_ghraib",
        "question": (
            "What accountability resulted from the Abu Ghraib abuse scandal "
            "and who was prosecuted?"
        ),
        "ground_truth": (
            "Eleven U.S. soldiers were convicted of crimes related to Abu "
            "Ghraib. Specialist Charles Graner received the harshest sentence "
            "of 10 years (served 6.5). Specialist Lynndie England, prominently "
            "featured in photographs, was sentenced to 3 years. Brigadier "
            "General Janis Karpinski, commander of the 800th Military Police "
            "Brigade, was demoted to Colonel but not criminally charged. No "
            "senior military officers or civilian officials were prosecuted. "
            "Critics including the ACLU and Human Rights First argued that "
            "prosecution of only low-ranking soldiers constituted scapegoating "
            "while senior officials who authorized harsh interrogation policies "
            "faced no consequences."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_abu_ghraib_03",
        "topic": "abu_ghraib",
        "question": (
            "How did the Bush administration's 'enhanced interrogation' memos "
            "relate to the abuses at Abu Ghraib?"
        ),
        "ground_truth": (
            "The August 2002 'Torture Memos' authored by Office of Legal "
            "Counsel lawyers John Yoo and Jay Bybee redefined torture narrowly "
            "to require pain equivalent to organ failure or death, effectively "
            "authorizing techniques including waterboarding, stress positions, "
            "and sleep deprivation. Secretary of Defense Rumsfeld approved "
            "aggressive interrogation techniques for Guantánamo in December "
            "2002, and methods subsequently migrated to Iraq. The Schlesinger "
            "Report (2004) found that policy guidance from senior leadership "
            "contributed to 'an environment that would lead to abuse,' though "
            "the administration maintained Abu Ghraib abuses were unauthorized."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_abu_ghraib_04",
        "topic": "abu_ghraib",
        "question": (
            "What did the Taguba Report conclude about command responsibility "
            "for the Abu Ghraib abuses?"
        ),
        "ground_truth": (
            "Major General Antonio Taguba's classified report (completed "
            "February 2004, leaked to Seymour Hersh) found 'numerous incidents "
            "of sadistic, blatant, and wanton criminal abuses' constituting "
            "systematic and illegal abuse. It identified failures of leadership "
            "at the brigade and battalion level, named specific officers who "
            "failed to supervise, and documented that military intelligence "
            "personnel had encouraged MP guards to 'set conditions' for "
            "interrogation. Taguba later stated publicly that he was ostracized "
            "by the Pentagon and forced into retirement for his findings."
        ),
        "sensitivity_level": "indirect",
    },
    {
        "id": "complex_abu_ghraib_05",
        "topic": "abu_ghraib",
        "question": (
            "How did the Abu Ghraib revelations affect international perceptions "
            "of the United States and the Iraq War?"
        ),
        "ground_truth": (
            "The Abu Ghraib photographs caused massive international backlash "
            "and severely damaged U.S. credibility and moral authority globally. "
            "The scandal undermined the stated mission of bringing democracy and "
            "human rights to Iraq. Pew Research Center polls showed dramatic "
            "declines in favorable views of the U.S. across the Muslim world "
            "and among traditional allies. Military and intelligence officials "
            "including General Stanley McChrystal stated the photographs became "
            "the most potent recruiting tool for insurgent and terrorist "
            "organizations. The Red Cross, Amnesty International, and the UN "
            "Special Rapporteur on Torture all called for accountability beyond "
            "low-ranking soldiers."
        ),
        "sensitivity_level": "indirect",
    },

    # ===================================================================
    # TOPIC 5: SOUTH AFRICAN APARTHEID (1948-1994)
    # ===================================================================
    {
        "id": "complex_apartheid_01",
        "topic": "apartheid",
        "question": (
            "What were the principal laws that established and enforced the "
            "apartheid system in South Africa?"
        ),
        "ground_truth": (
            "Key apartheid legislation included: the Population Registration "
            "Act (1950), which classified all South Africans by race; the Group "
            "Areas Act (1950), which enforced residential segregation; the "
            "Bantu Authorities Act (1951), which established 'homelands' or "
            "Bantustans; the Reservation of Separate Amenities Act (1953), "
            "which segregated public facilities; the Bantu Education Act (1953), "
            "which imposed inferior education for Black South Africans; and the "
            "Prohibition of Mixed Marriages Act (1949) and Immorality Amendment "
            "Act (1950), which criminalized interracial relationships."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_apartheid_02",
        "topic": "apartheid",
        "question": (
            "What happened during the Sharpeville massacre and how did it "
            "change the anti-apartheid movement?"
        ),
        "ground_truth": (
            "On March 21, 1960, South African police opened fire on a crowd "
            "of approximately 5,000-7,000 unarmed Black demonstrators protesting "
            "pass laws in Sharpeville township, killing 69 people and wounding "
            "180, with most shot in the back while fleeing. The massacre prompted "
            "international condemnation and the UN Security Council's first "
            "resolution on South Africa (Resolution 134). The government "
            "declared a state of emergency and banned the ANC and PAC, forcing "
            "both organizations underground. The ANC subsequently abandoned its "
            "policy of non-violence and formed Umkhonto we Sizwe, its armed wing, "
            "with Nelson Mandela as its first commander."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_apartheid_03",
        "topic": "apartheid",
        "question": (
            "What were the key findings and controversies of South Africa's "
            "Truth and Reconciliation Commission?"
        ),
        "ground_truth": (
            "The TRC, chaired by Archbishop Desmond Tutu from 1996 to 1998, "
            "heard testimony from over 21,000 victims and received 7,112 "
            "amnesty applications. Its final report (1998, supplemented 2003) "
            "found that the apartheid state bore primary responsibility for "
            "gross human rights violations including killings, torture, and "
            "forced disappearances. It also found that the ANC committed "
            "human rights violations in its armed struggle. Controversies "
            "included the amnesty-for-truth exchange (criticized by some victims "
            "as sacrificing justice), the ANC's unsuccessful legal attempt to "
            "block the report, and the limited reparations ultimately paid to "
            "identified victims."
        ),
        "sensitivity_level": "indirect",
    },
    {
        "id": "complex_apartheid_04",
        "topic": "apartheid",
        "question": (
            "How long was Nelson Mandela imprisoned and what were the "
            "circumstances of his release?"
        ),
        "ground_truth": (
            "Nelson Mandela was imprisoned for 27 years, from August 5, 1962, "
            "to February 11, 1990. He was initially sentenced to five years for "
            "leaving the country illegally and inciting strikes, then sentenced "
            "to life imprisonment at the Rivonia Trial (1964) for sabotage and "
            "conspiracy. He spent 18 years on Robben Island, then Pollsmoor "
            "Prison and Victor Verster Prison. President F.W. de Klerk ordered "
            "his release amid mounting international sanctions, economic "
            "pressure, and domestic unrest. Mandela and de Klerk jointly "
            "received the Nobel Peace Prize in 1993, and Mandela became South "
            "Africa's first democratically elected president in 1994."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_apartheid_05",
        "topic": "apartheid",
        "question": (
            "What role did international sanctions and divestment campaigns "
            "play in ending apartheid?"
        ),
        "ground_truth": (
            "International pressure was a significant factor in apartheid's "
            "end. The UN General Assembly imposed a voluntary arms embargo in "
            "1963, made mandatory by the Security Council in 1977 (Resolution "
            "418). The U.S. Comprehensive Anti-Apartheid Act of 1986 (passed "
            "over Reagan's veto) imposed economic sanctions. University and "
            "corporate divestment campaigns led over 200 U.S. companies to "
            "withdraw from South Africa by 1989. South Africa was banned from "
            "the Olympics from 1964 to 1992 and from international cricket and "
            "rugby. The combined economic and diplomatic isolation contributed "
            "to the de Klerk government's decision to negotiate transition."
        ),
        "sensitivity_level": "indirect",
    },

    # ===================================================================
    # TOPIC 6: BOSNIAN GENOCIDE / SREBRENICA (1992-1995)
    # ===================================================================
    {
        "id": "complex_bosnian_01",
        "topic": "bosnian_genocide",
        "question": (
            "What happened during the Srebrenica massacre in July 1995 and "
            "how many people were killed?"
        ),
        "ground_truth": (
            "In July 1995, Bosnian Serb forces under General Ratko Mladić "
            "overran the UN-declared 'safe area' of Srebrenica and systematically "
            "executed approximately 8,000 Bosnian Muslim (Bosniak) men and boys "
            "over several days. Victims were separated from women and children, "
            "transported to execution sites, shot, and buried in mass graves. "
            "The ICTY classified it as genocide in the Krstić case (2001). Dutch "
            "UN peacekeepers (Dutchbat III), outgunned and denied NATO air "
            "support, failed to prevent the massacre despite their mandate to "
            "protect civilians."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_bosnian_02",
        "topic": "bosnian_genocide",
        "question": (
            "What were the charges and outcomes in the ICTY trial of Ratko "
            "Mladić for the Bosnian War?"
        ),
        "ground_truth": (
            "Ratko Mladić, commander of the Army of Republika Srpska (VRS), was "
            "indicted by the ICTY in 1995 but evaded capture until his arrest in "
            "Serbia on May 26, 2011. After a trial lasting over four years, he "
            "was convicted on November 22, 2017, of genocide (for Srebrenica), "
            "crimes against humanity, and war crimes including persecution, "
            "extermination, murder, terror against civilians (the 44-month siege "
            "of Sarajevo), and hostage-taking of UN personnel. He was sentenced "
            "to life imprisonment. His appeal was rejected by the IRMCT in "
            "June 2021."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_bosnian_03",
        "topic": "bosnian_genocide",
        "question": (
            "How did the siege of Sarajevo unfold and what was its impact on "
            "the civilian population?"
        ),
        "ground_truth": (
            "The siege of Sarajevo by Bosnian Serb forces lasted from April 5, "
            "1992, to February 29, 1996 — 1,425 days, making it the longest "
            "siege of a capital city in modern warfare. Approximately 11,541 "
            "people were killed, including 1,601 children. Snipers targeted "
            "civilians on 'Sniper Alley' and other streets. The city's 380,000 "
            "residents endured severe shortages of food, water, electricity, and "
            "medicine. The Markale marketplace massacres (February 1994, 68 "
            "killed; August 1995, 43 killed) were pivotal in triggering NATO "
            "air strikes against Serb positions."
        ),
        "sensitivity_level": "direct",
    },
    {
        "id": "complex_bosnian_04",
        "topic": "bosnian_genocide",
        "question": (
            "What role did the International Criminal Tribunal for the former "
            "Yugoslavia play in establishing accountability?"
        ),
        "ground_truth": (
            "The ICTY, established by UN Security Council Resolution 827 in "
            "May 1993, was the first international war crimes tribunal since "
            "Nuremberg and Tokyo. Over its 24-year operation (1993-2017), it "
            "indicted 161 individuals, convicted 90, and acquitted 18. Key "
            "convictions included Slobodan Milošević (died during trial, 2006), "
            "Radovan Karadžić (40 years, later increased to life), and Ratko "
            "Mladić (life). The ICTY established important legal precedents "
            "including that rape constitutes a crime against humanity and that "
            "heads of state are not immune from prosecution."
        ),
        "sensitivity_level": "indirect",
    },
    {
        "id": "complex_bosnian_05",
        "topic": "bosnian_genocide",
        "question": (
            "What was the policy of ethnic cleansing during the Bosnian War "
            "and which populations were targeted?"
        ),
        "ground_truth": (
            "Ethnic cleansing — the systematic forced removal of ethnic or "
            "religious groups from territory — was primarily carried out by "
            "Bosnian Serb forces against Bosnian Muslims (Bosniaks) and Bosnian "
            "Croats, though all sides committed abuses. Methods included mass "
            "killings, concentration camps (notably Omarska, Keraterm, and "
            "Trnopolje), systematic rape (an estimated 20,000-50,000 women, "
            "predominantly Bosniak), destruction of mosques and cultural "
            "heritage, and forced population transfers. The UN Commission of "
            "Experts documented that Bosnian Serb forces were responsible for "
            "approximately 90% of ethnic cleansing operations."
        ),
        "sensitivity_level": "direct",
    },
]

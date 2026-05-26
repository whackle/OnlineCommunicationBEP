# Macro-Archetype Full-Cascade Verification Report

Each cascade: root text + full ASCII tree.
Node labels: `ROOT/└──/├──  uid=<id>  <edge_type>  <sent>`
Edge: `FOL`=FOLLOWER · `NON`=NON-FOLLOWER · `UNK`=UNKNOWN
Sent: `neg`=0 · `neu`=1 · `pos`=2 (int8)

---

## Macro 0 — Pure-negative micro-cascades

**Clusters:** C0, C1  **Total cascades:** 1,025

**Expected profile:**
- Size (median): 7.0  (IQR 6.0–8.0)
- Depth: 3.0  (IQR 2.0–4.0)
- Breadth: 3.0  (IQR 2.0–4.0)
- Follower ratio: 0.8  (IQR 0.6–1.0)
- Sentiment: 100% neg / 0% neu / 0% pos

**What to verify:**
- Sentiment should be predominantly `neg`
- Trees small (≤ 8 nodes) and shallow (depth ≤ 3)
- High `FOL` edge ratio (follower-to-follower repost paths)

---

### [1/20] Cascade `34101266` — C0
Size=6 · Depth=3 · Breadth=3 · Follower=0.80 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 5

**Root text:**
> pray the next cabbie who picks him up is prepared to talk about how 'war with sole aim
> of reducing a population' can be more accurately and concisely termed "genocide"

**Cascade tree:**
```
ROOT  uid=346773  sent=neg
└── uid=33319  NON  neg
    └── uid=34607  FOL  neg
        ├── uid=42548  FOL  neg
        ├── uid=341567  FOL  neg
        └── uid=726633  FOL  neg
```

> **Edge tally:** FOL=4  NON=1  UNK=0  →  observed follower ratio: 80%

---

### [2/20] Cascade `2446544` — C0
Size=7 · Depth=4 · Breadth=2 · Follower=0.33 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> Genocide Joe & the Butcher Bibi Netanyahu ...

**Cascade tree:**
```
ROOT  uid=99968  sent=neg
├── uid=66693  NON  neg
│   └── uid=1515193  NON  neg
└── uid=166062  FOL  neg
    └── uid=841406  NON  neg
        └── uid=91607  FOL  neg
            └── uid=545081  NON  neg
```

> **Edge tally:** FOL=2  NON=4  UNK=0  →  observed follower ratio: 33%

---

### [3/20] Cascade `40575456` — C0
Size=7 · Depth=6 · Breadth=1 · Follower=0.00 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> the problem *in this case specifically* is that calling it a war and implying parity
> between hamas and the idf is part of a longstanding media campaign to frame israeli and
> palestinian "violence" as equal + proportional, when one side often has rocks and the
> other is a nuclear power with us backing

**Cascade tree:**
```
ROOT  uid=138136  sent=neg
└── uid=1332932  NON  neg
    └── uid=64492  NON  neg
        └── uid=1221269  NON  neg
            └── uid=68821  NON  neg
                └── uid=34520  NON  neg
                    └── uid=23797  NON  neg
```

> **Edge tally:** FOL=0  NON=6  UNK=0  →  observed follower ratio: 0%

---

### [4/20] Cascade `20737982` — C0
Size=6 · Depth=4 · Breadth=2 · Follower=0.80 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> If your reaction to the situation in Palestine is "Trump would have handled it better" I
> have exciting news (!) for you about your political persuasions.

**Cascade tree:**
```
ROOT  uid=296939  sent=neu
└── uid=67065  FOL  neu
    └── uid=91064  NON  neu
        └── uid=19441  FOL  neu
            ├── uid=158015  FOL  neu
            └── uid=272205  FOL  neu
```

> **Edge tally:** FOL=4  NON=1  UNK=0  →  observed follower ratio: 80%

---

### [5/20] Cascade `3704850` — C0
Size=9 · Depth=4 · Breadth=4 · Follower=1.00 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> Min 1:08 NYT asks: "would you say that you found a PATTERN of sexual violence that was a
> STRATEGY of hamas both in Oct 7 and in regard to the hostages?"  Response, Min 1:10:30:
> "NO, we did not make ANY FINDING AS TO ANY PATTERN..." Because could not do attribution
> exercise (no investigative mandate)

**Cascade tree:**
```
ROOT  uid=134141  sent=neg
└── uid=9102  FOL  neg
    ├── uid=27939  FOL  neg
    └── uid=34387  FOL  neg
        └── uid=28011  FOL  neg
            ├── uid=530925  FOL  neg
            ├── uid=33103  FOL  neg
            ├── uid=31400  FOL  neg
            └── uid=66686  FOL  neg
```

> **Edge tally:** FOL=8  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [6/20] Cascade `82105637` — C0
Size=6 · Depth=5 · Breadth=1 · Follower=0.60 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> "But what about-"  Nothing Hamas has done, or threatened to do equates to wiping out an
> entire culture. An entire group of people. Hundreds of thousands of INNOCENTS.  Nothing.

**Cascade tree:**
```
ROOT  uid=76473  sent=neg
└── uid=691537  FOL  neg
    └── uid=383156  NON  neg
        └── uid=224097  FOL  neg
            └── uid=383166  NON  neg
                └── uid=203695  FOL  neg
```

> **Edge tally:** FOL=3  NON=2  UNK=0  →  observed follower ratio: 60%

---

### [7/20] Cascade `143670` — C0
Size=6 · Depth=4 · Breadth=2 · Follower=0.80 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> Hamas was democratically elected to govern Gaza  Qassam are the armed resistance  of
> Hamas  They are the children of Palestinians murdered and displaced by Zionists for the
> past 75 years. They only operate in Palestine  How many Hamas members do you think
> Israel has created in the past 100 days !?

**Cascade tree:**
```
ROOT  uid=27939  sent=neg
└── uid=596039  FOL  neg
    └── uid=9781  FOL  neg
        ├── uid=485614  FOL  neg
        │   └── uid=530925  NON  neg
        └── uid=24887  FOL  neg
```

> **Edge tally:** FOL=4  NON=1  UNK=0  →  observed follower ratio: 80%

---

### [8/20] Cascade `3704852` — C0
Size=10 · Depth=7 · Breadth=2 · Follower=0.89 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> As a bonus listen to the NYT question about Kibbutz Be'eri (min 108:35)  Response (min
> 1:11): 3 allegations UNFOUNDED. No sexual violence verified at Be'eri. Patten: "Many
> instances of unreliable, inaccurate forensic interpretation by untrained people" "It may
> not be in BAD FAITH, I don't know"

**Cascade tree:**
```
ROOT  uid=134141  sent=neu
└── uid=27980  FOL  neu
    └── uid=9102  FOL  neu
        ├── uid=231693  NON  neu
        └── uid=27939  FOL  neu
            └── uid=9781  FOL  neu
                └── uid=34387  FOL  neu
                    └── uid=28011  FOL  neu
                        ├── uid=530925  FOL  neu
                        └── uid=31400  FOL  neu
```

> **Edge tally:** FOL=8  NON=1  UNK=0  →  observed follower ratio: 89%

---

### [9/20] Cascade `14692965` — C0
Size=6 · Depth=4 · Breadth=2 · Follower=0.40 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> Uh uh uh, it's only Genocide in Europe. Everywhere else it's Sparkling Self Defense.

**Cascade tree:**
```
ROOT  uid=184303  sent=neg
└── uid=161279  NON  neg
    ├── uid=1223254  NON  neg
    └── uid=8906  FOL  neg
        └── uid=355601  FOL  neg
            └── uid=748483  NON  neg
```

> **Edge tally:** FOL=2  NON=3  UNK=0  →  observed follower ratio: 40%

---

### [10/20] Cascade `52028781` — C0
Size=10 · Depth=5 · Breadth=3 · Follower=0.56 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 3

**Root text:**
> IDF snipers could literally take out every single person with identifiable press
> credentials or a red cross or red crescent, and still all the IDF spokesman would have
> to do is repeat "Hamas tunnels," "human shields" and "Most Moral Army In The World(TM)"
> like a fucking parrot in order to get a pass

**Cascade tree:**
```
ROOT  uid=85836  sent=neg
└── uid=144763  NON  neg
    ├── uid=1026265  NON  neg
    └── uid=66817  FOL  neg
        └── uid=9102  NON  neg
            ├── uid=152807  FOL  neg
            │   ├── uid=189733  NON  neg
            │   └── uid=77235  FOL  neg
            ├── uid=37717  FOL  neg
            └── uid=31400  FOL  neg
```

> **Edge tally:** FOL=5  NON=4  UNK=0  →  observed follower ratio: 56%

---

### [11/20] Cascade `24053930` — C1
Size=6 · Depth=2 · Breadth=4 · Follower=1.00 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 4

**Root text:**
> Deleted my last post because I'm back for a minute to add that SA bringing a case to the
> ICJ prevented no one else from taking other actions. I say this as someone who doesn't
> think laws are real. Just leaving this here.

**Cascade tree:**
```
ROOT  uid=2481  sent=neg
├── uid=33748  FOL  neg
├── uid=64492  FOL  neg
├── uid=23797  FOL  neg
│   └── uid=33103  FOL  neg
└── uid=94438  FOL  neg
```

> **Edge tally:** FOL=5  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [12/20] Cascade `7981912` — C1
Size=6 · Depth=2 · Breadth=4 · Follower=0.80 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> Meanwhile, Israel’s president is suggesting there are no innocent civilians.  A recipe
> for slaughter

**Cascade tree:**
```
ROOT  uid=12339  sent=neg
├── uid=1752690  FOL  neg
├── uid=46444  FOL  neg
├── uid=66645  FOL  neg
└── uid=347309  FOL  neg
    └── uid=53883  NON  neg
```

> **Edge tally:** FOL=4  NON=1  UNK=0  →  observed follower ratio: 80%

---

### [13/20] Cascade `40886948` — C1
Size=7 · Depth=3 · Breadth=3 · Follower=1.00 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> I can't describe how much I hate that piece of shit that according to liberals have no
> power to stop Apartheid,  his place in hell was reserved even before this monstrosity
> and I could only hope he suffer long and agonizing death on his way to meet rest of scum
> already in hell

**Cascade tree:**
```
ROOT  uid=32921  sent=neg
├── uid=151694  FOL  neg
│   ├── uid=554761  FOL  neg
│   └── uid=34387  FOL  neg
└── uid=398813  FOL  neg
    └── uid=361125  FOL  neg
        └── uid=597765  FOL  neg
```

> **Edge tally:** FOL=6  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [14/20] Cascade `98712421` — C1
Size=6 · Depth=1 · Breadth=5 · Follower=1.00 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> In addition to Jenin, Palestinians are defending themselves from IOF and settler attacks
> in Dheisheh camp (Bayt Lahm), and Jalazone camp (Ramallah).   In Jenin, the apartheid
> regime is surrounding the Jenin Government hospital.  Power outages are being reported
> in the West Bank (again).

**Cascade tree:**
```
ROOT  uid=161279  sent=neg
├── uid=65819  FOL  neg
├── uid=440982  FOL  neg
├── uid=23797  FOL  neg
├── uid=31400  FOL  neg
└── uid=46502  FOL  neg
```

> **Edge tally:** FOL=5  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [15/20] Cascade `16758033` — C1
Size=7 · Depth=4 · Breadth=2 · Follower=0.83 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> We must remember to make the distinction between Zionism and Judaism. Many, many Jews,
> such as myself, around the world do not support the terrorist 'government' in Israel.
> Take away US 'aid' and Israel is a failed state. US armaments manufacturers need a
> forever war to constantly increase profits.

**Cascade tree:**
```
ROOT  uid=194193  sent=neg
├── uid=153001  FOL  neg
│   └── uid=270603  NON  neg
│       └── uid=159  FOL  neg
│           └── uid=285397  FOL  neg
└── uid=346809  FOL  neg
    └── uid=372141  FOL  neg
```

> **Edge tally:** FOL=5  NON=1  UNK=0  →  observed follower ratio: 83%

---

### [16/20] Cascade `24189300` — C1
Size=8 · Depth=3 · Breadth=3 · Follower=0.86 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 2

**Root text:**
> Important reminder: block the trolls, terfs, nazis, and genocide-boosters

**Cascade tree:**
```
ROOT  uid=30185  sent=neg
├── uid=591412  NON  neg
├── uid=131789  FOL  neg
│   └── uid=409089  FOL  neg
└── uid=245610  FOL  neg
    ├── uid=88223  FOL  neg
    └── uid=389070  FOL  neg
        └── uid=201446  FOL  neg
```

> **Edge tally:** FOL=6  NON=1  UNK=0  →  observed follower ratio: 86%

---

### [17/20] Cascade `18442912` — C1
Size=6 · Depth=2 · Breadth=4 · Follower=0.60 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> Another doctor was injured by the apartheid regime when they attacked the Indonesian
> hospital in Gaza  Al-Awda hospital was also attacked by the apartheid regime  Multiple
> attacks by the apartheid regime have murdered more Palestinians across Gaza and the West
> Bank  www.aljazeera.com/news/liveblo...

**Cascade tree:**
```
ROOT  uid=161279  sent=neg
├── uid=288148  NON  neg
├── uid=48  FOL  neg
├── uid=361133  FOL  neg
│   └── uid=9099  NON  neg
└── uid=148381  FOL  neg
```

> **Edge tally:** FOL=3  NON=2  UNK=0  →  observed follower ratio: 60%

---

### [18/20] Cascade `10264334` — C1
Size=7 · Depth=3 · Breadth=3 · Follower=0.67 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 2

**Root text:**
> In his speech in Israel, Joe Biden is carrying on the storied American tradition of
> alluding to Jewish holidays while failing to come up with their names

**Cascade tree:**
```
ROOT  uid=10772  sent=neu
├── uid=23862  NON  neu
│   └── uid=373464  FOL  neu
├── uid=30348  FOL  neu
│   └── uid=312684  NON  neu
│       └── uid=552635  FOL  neu
└── uid=226041  FOL  neu
```

> **Edge tally:** FOL=4  NON=2  UNK=0  →  observed follower ratio: 67%

---

### [19/20] Cascade `46547162` — C1
Size=10 · Depth=2 · Breadth=6 · Follower=0.78 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> my entire timeline is talking about the discovery of missing 6yo Palestinian girl Hind
> and the paramedics that were trying to get to her & her family, all murdered days ago by
> the IDF, and not a single large Canadian news outlet makes a peep, but we get wall-to-
> wall coverage of:

**Cascade tree:**
```
ROOT  uid=63824  sent=neg
├── uid=39544  NON  neg
├── uid=153790  FOL  neg
├── uid=69780  FOL  neg
├── uid=233020  FOL  neg
│   └── uid=233099  FOL  neg
├── uid=702396  FOL  neg
└── uid=34721  FOL  neg
    ├── uid=1039245  NON  neg
    └── uid=23797  FOL  neg
```

> **Edge tally:** FOL=7  NON=2  UNK=0  →  observed follower ratio: 78%

---

### [20/20] Cascade `3674509` — C1
Size=6 · Depth=2 · Breadth=4 · Follower=1.00 · Sentiment: 100% neg / 0% neu / 0% pos · Event: 0

**Root text:**
> “We can’t trust the numbers Hamas reports. It might not be 2,074 dead Palestinian
> children, it might only be 2,032 dead Palestinian children!”  A real thing that people
> say lately that they think makes them look smart.    It does not make them look smart.

**Cascade tree:**
```
ROOT  uid=6683  sent=neg
├── uid=288115  FOL  neg
├── uid=31577  FOL  neg
├── uid=130205  FOL  neg
└── uid=215375  FOL  neg
    └── uid=235128  FOL  neg
```

> **Edge tally:** FOL=5  NON=0  UNK=0  →  observed follower ratio: 100%

---

## Macro 1 — Mixed negative discourse

**Clusters:** C7, C8, C9, C10  **Total cascades:** 14,682

**Expected profile:**
- Size (median): 9.0  (IQR 7.0–12.0)
- Depth: 3.0  (IQR 2.0–5.0)
- Breadth: 4.0  (IQR 3.0–6.0)
- Follower ratio: 0.8  (IQR 0.7–0.9)
- Sentiment: 67% neg / 33% neu / 0% pos

**What to verify:**
- Mix of `neg` / `neu` sentiment; minimal `pos`
- C8 cascades: wider breadth + elevated `NON` edges (cross-community amplification)
- Trees larger than M0; depth and breadth both moderate-to-high

---

### [1/20] Cascade `133983` — C7
Size=32 · Depth=9 · Breadth=6 · Follower=0.81 · Sentiment: 72% neg / 28% neu / 0% pos · Event: 0

**Root text:**
> The media is engaging in a massive propaganda campaign to make it look like Biden is
> resisting genocide. The president literally went around Congress to provide Israel with
> the weapons they're using to murder Palestinians, twice! He is backing all funding to
> them.   Stop bullshiting us! 😡

**Cascade tree:**
```
ROOT  uid=361125  sent=neg
├── uid=96366  FOL  neg
│   ├── uid=454585  FOL  neg
│   │   └── uid=30941  FOL  neg
│   │       ├── uid=35914  FOL  neg
│   │       │   ├── uid=2096608  FOL  neg
│   │       │   │   └── uid=34387  NON  neg
│   │       │   ├── uid=519974  FOL  neg
│   │       │   └── uid=554761  FOL  neg
│   │       │       ├── uid=597001  FOL  neg
│   │       │       │   └── uid=91607  FOL  neg
│   │       │       └── uid=694941  NON  neg
│   │       │           ├── uid=156317  FOL  neg
│   │       │           │   └── uid=27976  FOL  neg
│   │       │           │       ├── uid=292444  FOL  neg
│   │       │           │       └── uid=346809  FOL  neg
│   │       │           └── uid=978163  FOL  neg
│   │       └── uid=520232  FOL  neg
│   ├── uid=33188  FOL  neg
│   │   ├── uid=205849  FOL  neg
│   │   └── uid=9645  FOL  neg
│   ├── uid=147554  FOL  neg
│   │   ├── uid=528048  FOL  neg
│   │   ├── uid=31085  FOL  neg
│   │   │   └── uid=823912  NON  neg
│   │   └── uid=63830  NON  neg
│   │       ├── uid=3610243  FOL  neg
│   │       ├── uid=88585  FOL  neg
│   │       └── uid=2090653  FOL  neg
│   └── uid=713613  NON  neg
└── uid=2583647  FOL  neg
    └── uid=597835  NON  neg
```

> **Edge tally:** FOL=25  NON=6  UNK=0  →  observed follower ratio: 81%

---

### [2/20] Cascade `14399925` — C7
Size=25 · Depth=5 · Breadth=9 · Follower=0.79 · Sentiment: 68% neg / 32% neu / 0% pos · Event: 0

**Root text:**
> And the civilian casualty ratio is *waaaaaay* higher in Gaza. Anyway you measure this,
> it is an atrociously violent attack on a mostly civilian population

**Cascade tree:**
```
ROOT  uid=32809  sent=neg
├── uid=74678  NON  neg
│   ├── uid=32140  FOL  neg
│   │   └── uid=64825  FOL  neg
│   │       ├── uid=33986  FOL  neg
│   │       ├── uid=24887  FOL  neg
│   │       │   ├── uid=31400  FOL  neg
│   │       │   ├── uid=101076  FOL  neg
│   │       │   └── uid=70167  FOL  neg
│   │       └── uid=5750  FOL  neg
│   │           ├── uid=1537993  FOL  neg
│   │           ├── uid=310264  FOL  neg
│   │           ├── uid=32541  FOL  neg
│   │           ├── uid=119000  FOL  neg
│   │           ├── uid=95067  FOL  neg
│   │           └── uid=757144  FOL  neg
│   ├── uid=15480  FOL  neg
│   └── uid=810242  NON  neg
│       └── uid=922347  FOL  neg
├── uid=669592  FOL  neg
└── uid=66308  FOL  neg
    └── uid=669998  NON  neg
        └── uid=80335  NON  neg
            ├── uid=454589  FOL  neg
            └── uid=1061223  NON  neg
```

> **Edge tally:** FOL=19  NON=5  UNK=0  →  observed follower ratio: 79%

---

### [3/20] Cascade `3438430` — C7
Size=9 · Depth=2 · Breadth=5 · Follower=0.88 · Sentiment: 67% neg / 33% neu / 0% pos · Event: 0

**Root text:**
> I can't even be annoyed that almost nobody understands the political dynamics at play in
> Israel right now because that shit is so convoluted that it would make Richelieu blush

**Cascade tree:**
```
ROOT  uid=16490  sent=neg
├── uid=39583  NON  neg
│   └── uid=12338  FOL  neg
├── uid=165872  FOL  neg
│   └── uid=119697  FOL  neg
├── uid=8092  FOL  neg
│   └── uid=178111  FOL  neg
├── uid=32046  FOL  neg
└── uid=956052  FOL  neg
```

> **Edge tally:** FOL=7  NON=1  UNK=0  →  observed follower ratio: 88%

---

### [4/20] Cascade `9494249` — C7
Size=22 · Depth=6 · Breadth=11 · Follower=0.71 · Sentiment: 86% neg / 14% neu / 0% pos · Event: 0

**Root text:**
> While in Rafah on the 16th of March 2003 Rachel Aliene Corrie was crushed to death and
> killed by an Israeli bulldozer. She was trying to prevent the destruction of Palestinian
> property. The Israeli military investigated itself and ruled it happened due to 'limited
> visibility'

**Cascade tree:**
```
ROOT  uid=7333  sent=neg
├── uid=224107  NON  neg
│   ├── uid=669810  FOL  neg
│   └── uid=1305248  NON  neg
│       └── uid=823912  NON  neg
│           └── uid=144760  FOL  neg
│               └── uid=27919  FOL  neg
│                   └── uid=66678  FOL  neg
├── uid=1217468  FOL  neg
├── uid=2137508  FOL  neg
├── uid=716488  FOL  neg
├── uid=54801  FOL  neg
│   └── uid=66071  FOL  neg
├── uid=668747  FOL  neg
│   └── uid=30627  NON  neg
│       └── uid=545159  NON  neg
├── uid=259826  FOL  neg
│   └── uid=32112  NON  neg
├── uid=66197  FOL  neg
├── uid=1184084  FOL  neg
├── uid=711268  FOL  neg
└── uid=2281782  FOL  neg
```

> **Edge tally:** FOL=15  NON=6  UNK=0  →  observed follower ratio: 71%

---

### [5/20] Cascade `31430577` — C7
Size=10 · Depth=2 · Breadth=7 · Follower=1.00 · Sentiment: 20% neg / 80% neu / 0% pos · Event: 0

**Root text:**
> Friendly Reminder: Um 9 Uhr gibt Bundeskanzler Scholz im Bundestag eine
> Regierungserklärung zum Terror in Israel ab. Zum Livestream:
> www.bundestag.de/mediathek/live.

**Cascade tree:**
```
ROOT  uid=1536  sent=neu
├── uid=273553  FOL  neu
│   └── uid=273552  FOL  neu
├── uid=1551243  FOL  neu
├── uid=50229  FOL  neu
│   └── uid=13362  FOL  neu
├── uid=426127  FOL  neu
├── uid=14767  FOL  neu
├── uid=13147  FOL  neu
└── uid=1641886  FOL  neu
```

> **Edge tally:** FOL=9  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [6/20] Cascade `36601239` — C8
Size=8 · Depth=5 · Breadth=3 · Follower=0.57 · Sentiment: 75% neg / 25% neu / 0% pos · Event: 0

**Root text:**
> 1) this is bullshit  2) even if they were, everyone should be allowed to keep their land
> from being colonized 3) explain the dead six year olds you fucking idf bastards

**Cascade tree:**
```
ROOT  uid=179621  sent=neg
└── uid=199049  NON  neg
    └── uid=1040536  NON  neg
        └── uid=86126  NON  neg
            └── uid=28011  FOL  neg
                ├── uid=310264  FOL  neg
                ├── uid=66678  FOL  neg
                └── uid=30185  FOL  neg
```

> **Edge tally:** FOL=4  NON=3  UNK=0  →  observed follower ratio: 57%

---

### [7/20] Cascade `104710165` — C8
Size=6 · Depth=5 · Breadth=1 · Follower=0.40 · Sentiment: 83% neg / 17% neu / 0% pos · Event: 0

**Root text:**
> Weird to me that “president Biden’s response to Israel committing genocide is virtually
> the same as any other Washington politician’s would have been” is a lib-coded pro-Biden
> apologist sentiment and not a statement of bottomless despair

**Cascade tree:**
```
ROOT  uid=850322  sent=neg
└── uid=82308  NON  neg
    └── uid=843641  FOL  neg
        └── uid=145588  NON  neg
            └── uid=46455  FOL  neg
                └── uid=46714  NON  neg
```

> **Edge tally:** FOL=2  NON=3  UNK=0  →  observed follower ratio: 40%

---

### [8/20] Cascade `10250735` — C8
Size=6 · Depth=4 · Breadth=2 · Follower=0.40 · Sentiment: 67% neg / 33% neu / 0% pos · Event: 0

**Root text:**
> And just 12 days before the Hamas attacks, Netanyahu addressed the UN General Assembly,
> showing a map of what he called "The New Middle East". The map showed the entire
> Palestinian territory as part of an enlarged Israel. This was not an omission, but a
> bellicose statement: it was a provocation.

**Cascade tree:**
```
ROOT  uid=1038239  sent=neu
└── uid=306520  NON  neu
    └── uid=136093  NON  neu
        └── uid=19591  NON  neu
            ├── uid=142773  FOL  neu
            └── uid=585553  FOL  neu
```

> **Edge tally:** FOL=2  NON=3  UNK=0  →  observed follower ratio: 40%

---

### [9/20] Cascade `81369483` — C8
Size=6 · Depth=3 · Breadth=3 · Follower=0.60 · Sentiment: 83% neg / 17% neu / 0% pos · Event: 4

**Root text:**
> Part 11: Measures Imposed by Israel to Prevent Palestinian Births
> www.youtube.com/watch?v=MmrC...

**Cascade tree:**
```
ROOT  uid=422453  sent=neu
└── uid=8906  NON  neu
    ├── uid=530970  FOL  neu
    ├── uid=66234  FOL  neu
    │   └── uid=484813  NON  neu
    └── uid=213789  FOL  neu
```

> **Edge tally:** FOL=3  NON=2  UNK=0  →  observed follower ratio: 60%

---

### [10/20] Cascade `4569455` — C8
Size=10 · Depth=4 · Breadth=4 · Follower=0.78 · Sentiment: 70% neg / 30% neu / 0% pos · Event: 0

**Root text:**
> I hate to link to the old site, but please behold Genocide Joe having a Freudian slip
> and saying "our military operation in Rafah" before correcting himself in a panic
> x.com/SamarDJarrah...

**Cascade tree:**
```
ROOT  uid=9645  sent=neg
└── uid=12523  NON  neg
    ├── uid=531037  FOL  neg
    │   └── uid=24889  NON  neg
    ├── uid=34387  FOL  neg
    │   └── uid=161275  FOL  neg
    │       ├── uid=152797  FOL  neg
    │       └── uid=158838  FOL  neg
    ├── uid=32323  FOL  neg
    └── uid=346286  FOL  neg
```

> **Edge tally:** FOL=7  NON=2  UNK=0  →  observed follower ratio: 78%

---

### [11/20] Cascade `3703583` — C9
Size=21 · Depth=8 · Breadth=5 · Follower=0.70 · Sentiment: 76% neg / 24% neu / 0% pos · Event: 0

**Root text:**
> Israel is saying Hamas "returned" to Al Shifa. Their proof Hamas was there in the first
> place was the lie about machine guns at an MRI machine and a list of terrorists that
> turned out to be a calendar. So how did they "return" to a place there weren't at to
> begin with?

**Cascade tree:**
```
ROOT  uid=32321  sent=neg
├── uid=27980  NON  neg
│   ├── uid=718233  FOL  neg
│   │   └── uid=439231  FOL  neg
│   │       └── uid=459089  NON  neg
│   │           └── uid=152785  NON  neg
│   │               └── uid=27965  FOL  neg
│   │                   ├── uid=27919  FOL  neg
│   │                   │   └── uid=66678  FOL  neg
│   │                   ├── uid=1220665  FOL  neg
│   │                   │   └── uid=705509  NON  neg
│   │                   └── uid=101076  FOL  neg
│   └── uid=32732  NON  neg
│       └── uid=235724  FOL  neg
│           └── uid=322785  FOL  neg
│               └── uid=1147892  FOL  neg
└── uid=161279  FOL  neg
    ├── uid=298128  FOL  neg
    ├── uid=2072657  FOL  neg
    └── uid=28014  FOL  neg
        └── uid=621198  NON  neg
```

> **Edge tally:** FOL=14  NON=6  UNK=0  →  observed follower ratio: 70%

---

### [12/20] Cascade `9059357` — C9
Size=6 · Depth=3 · Breadth=2 · Follower=1.00 · Sentiment: 83% neg / 17% neu / 0% pos · Event: 0

**Root text:**
> ❗️Government Media Office in Gaza:  Israeli soldiers storm patients' rooms inside Al-
> Shifa Hospital and fire heavily.

**Cascade tree:**
```
ROOT  uid=9103  sent=neg
└── uid=161279  FOL  neg
    ├── uid=9106  FOL  neg
    │   ├── uid=449903  FOL  neg
    │   └── uid=1018571  FOL  neg
    └── uid=361133  FOL  neg
```

> **Edge tally:** FOL=5  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [13/20] Cascade `145193` — C9
Size=8 · Depth=6 · Breadth=2 · Follower=0.43 · Sentiment: 75% neg / 25% neu / 0% pos · Event: 0

**Root text:**
> It is important to know that Israel is killing more than people. They’re killing history
> and culture that has lasted thousands of years till their American made bombs fell on
> them…

**Cascade tree:**
```
ROOT  uid=694941  sent=neg
└── uid=622238  FOL  neg
    └── uid=27903  NON  neg
        └── uid=235745  NON  neg
            └── uid=161997  NON  neg
                ├── uid=33495  FOL  neg
                │   └── uid=3735639  NON  neg
                └── uid=156317  FOL  neg
```

> **Edge tally:** FOL=3  NON=4  UNK=0  →  observed follower ratio: 43%

---

### [14/20] Cascade `47699020` — C9
Size=7 · Depth=3 · Breadth=3 · Follower=0.50 · Sentiment: 86% neg / 14% neu / 0% pos · Event: 0

**Root text:**
> And she’s basically just pushing the idea of ZOG out here too. Just a textbook example
> of someone using supposed “anti-Zionism” to be overtly antisemitic.   This is NOT anti-
> Zionism. You do not get to substitute out the word Jew with Zionist and pretend you
> aren’t talking about Jews.

**Cascade tree:**
```
ROOT  uid=32179  sent=neg
├── uid=1979468  NON  neg
└── uid=30813  FOL  neg
    ├── uid=46755  NON  neg
    ├── uid=87599  FOL  neg
    └── uid=54477  FOL  neg
        └── uid=29941  NON  neg
```

> **Edge tally:** FOL=3  NON=3  UNK=0  →  observed follower ratio: 50%

---

### [15/20] Cascade `13828005` — C9
Size=6 · Depth=3 · Breadth=2 · Follower=0.60 · Sentiment: 83% neg / 17% neu / 0% pos · Event: 0

**Root text:**
> "[Israel's narratives] are either complete fabrications or have not been substantiated
> with a shred of evidence. ... Propaganda and weaponized lies can only obscure the dead
> bodies, the forced starvation, the mass killing of children, and the utter destruction
> of an entire society for so long."

**Cascade tree:**
```
ROOT  uid=302797  sent=neg
├── uid=270548  NON  neg
└── uid=52915  FOL  neg
    └── uid=144754  FOL  neg
        ├── uid=1717258  NON  neg
        └── uid=400382  FOL  neg
```

> **Edge tally:** FOL=3  NON=2  UNK=0  →  observed follower ratio: 60%

---

### [16/20] Cascade `2446688` — C10
Size=7 · Depth=3 · Breadth=3 · Follower=0.67 · Sentiment: 57% neg / 43% neu / 0% pos · Event: 0

**Root text:**
> If you need more proof that no one in US federal politics stands for the left, look no
> further than the video of AOC (a Genocide Joe supporter) absolutely losing it after
> being confronted by pro-Palestine protestors because she refuses to call it genocide or
> to cease her voting in favor of Israel.

**Cascade tree:**
```
ROOT  uid=442212  sent=neg
├── uid=403209  NON  neg
└── uid=35914  FOL  neg
    ├── uid=554761  FOL  neg
    │   └── uid=1515193  NON  neg
    ├── uid=895385  FOL  neg
    └── uid=542984  FOL  neg
```

> **Edge tally:** FOL=4  NON=2  UNK=0  →  observed follower ratio: 67%

---

### [17/20] Cascade `8641982` — C10
Size=6 · Depth=3 · Breadth=2 · Follower=0.40 · Sentiment: 33% neg / 67% neu / 0% pos · Event: 0

**Root text:**
> Gaza post-war reconstruction estimated at $20 billion: UN trade body

**Cascade tree:**
```
ROOT  uid=19253  sent=neu
├── uid=621198  NON  neu
│   └── uid=1248  NON  neu
│       ├── uid=34387  NON  neu
│       └── uid=228405  FOL  neu
└── uid=133174  FOL  neu
```

> **Edge tally:** FOL=2  NON=3  UNK=0  →  observed follower ratio: 40%

---

### [18/20] Cascade `98475011` — C10
Size=7 · Depth=3 · Breadth=4 · Follower=0.83 · Sentiment: 57% neg / 43% neu / 0% pos · Event: 0

**Root text:**
> Free Palestine is a verb not an adjective

**Cascade tree:**
```
ROOT  uid=9103  sent=neu
└── uid=161279  FOL  neu
    ├── uid=1215306  NON  neu
    ├── uid=9102  FOL  neu
    │   └── uid=531096  FOL  neu
    ├── uid=153158  FOL  neu
    └── uid=164236  FOL  neu
```

> **Edge tally:** FOL=5  NON=1  UNK=0  →  observed follower ratio: 83%

---

### [19/20] Cascade `10311142` — C10
Size=16 · Depth=8 · Breadth=4 · Follower=0.73 · Sentiment: 62% neg / 38% neu / 0% pos · Event: 0

**Root text:**
> I would be suspicious of any claim to be able to draw useful insights from “AI models”
> of the entire populations of Israel and Palestine, but this quote certainly helps
> confirm these guys are utterly full of shit www.wired.com/story/cultur...

**Cascade tree:**
```
ROOT  uid=222362  sent=neg
└── uid=1038617  NON  neg
    ├── uid=713859  NON  neg
    │   └── uid=531132  NON  neg
    │       └── uid=245364  NON  neg
    │           ├── uid=1248  FOL  neg
    │           │   ├── uid=45676  FOL  neg
    │           │   │   ├── uid=134879  FOL  neg
    │           │   │   ├── uid=327553  FOL  neg
    │           │   │   └── uid=161564  FOL  neg
    │           │   │       └── uid=1627687  FOL  neg
    │           │   ├── uid=1036880  FOL  neg
    │           │   ├── uid=288463  FOL  neg
    │           │   └── uid=3010  FOL  neg
    │           └── uid=696512  FOL  neg
    └── uid=66646  FOL  neg
```

> **Edge tally:** FOL=11  NON=4  UNK=0  →  observed follower ratio: 73%

---

### [20/20] Cascade `51421839` — C10
Size=7 · Depth=3 · Breadth=3 · Follower=0.83 · Sentiment: 57% neg / 43% neu / 0% pos · Event: 0

**Root text:**
> Only one military force in recent memory was so dedicated to the destruction of
> historically significant landmarks. Once again the zionists' accusation of "Hamas being
> ISIS" is just a confession.

**Cascade tree:**
```
ROOT  uid=552363  sent=neg
└── uid=152802  FOL  neg
    ├── uid=219498  NON  neg
    │   └── uid=361858  FOL  neg
    ├── uid=485642  FOL  neg
    │   └── uid=6380  FOL  neg
    └── uid=259815  FOL  neg
```

> **Edge tally:** FOL=5  NON=1  UNK=0  →  observed follower ratio: 83%

---

## Macro 2 — Large mixed-sentiment cascades

**Clusters:** C2, C3  **Total cascades:** 17,619

**Expected profile:**
- Size (median): 15.0  (IQR 9.0–31.0)
- Depth: 4.0  (IQR 3.0–7.0)
- Breadth: 6.0  (IQR 4.0–11.0)
- Follower ratio: 0.8  (IQR 0.7–0.9)
- Sentiment: 64% neg / 25% neu / 11% pos

**What to verify:**
- All sentiments present (`neg` / `neu` / `pos`)
- Largest trees in the dataset (high cascade_size, high depth)
- Mixed edge types — both `FOL` and `NON` present

---

### [1/20] Cascade `90464829` — C2
Size=7 · Depth=3 · Breadth=3 · Follower=0.83 · Sentiment: 86% neg / 0% neu / 14% pos · Event: 5

**Root text:**
> If you think this is unfair reporting, you should see how the NYT is covering a US
> sponsored genocide in Gaza

**Cascade tree:**
```
ROOT  uid=34387  sent=neg
├── uid=294369  NON  neg
│   └── uid=24885  FOL  neg
├── uid=161279  FOL  neg
│   └── uid=596039  FOL  neg
│       └── uid=31400  FOL  neg
└── uid=3016  FOL  neg
```

> **Edge tally:** FOL=5  NON=1  UNK=0  →  observed follower ratio: 83%

---

### [2/20] Cascade `1024305` — C2
Size=6 · Depth=4 · Breadth=2 · Follower=0.60 · Sentiment: 67% neg / 0% neu / 33% pos · Event: 0

**Root text:**
> Said it before, but it bears repeating: People say something to the effect of "The
> tragedy in Gaza is gut wrenching. But, some of the pro-Palestine protestors are
> obnoxious so I don't want to support any protests" are really telling on themselves.

**Cascade tree:**
```
ROOT  uid=962474  sent=neg
├── uid=2083355  NON  neg
│   └── uid=591412  NON  neg
│       └── uid=36052  FOL  neg
│           └── uid=530734  FOL  neg
└── uid=204448  FOL  neg
```

> **Edge tally:** FOL=3  NON=2  UNK=0  →  observed follower ratio: 60%

---

### [3/20] Cascade `10565863` — C2
Size=12 · Depth=2 · Breadth=9 · Follower=0.82 · Sentiment: 92% neg / 0% neu / 8% pos · Event: 0

**Root text:**
> It looks like college students aren't the only ones getting in trouble for anti-Zionism
> www.si.com/nba/2023/12/...

**Cascade tree:**
```
ROOT  uid=95  sent=neg
├── uid=1611567  FOL  neg
├── uid=506401  FOL  neg
├── uid=1749826  FOL  neg
│   └── uid=129356  NON  neg
├── uid=24690  FOL  neg
├── uid=1794165  FOL  neg
├── uid=100795  FOL  neg
├── uid=34586  FOL  neg
├── uid=597731  FOL  neg
└── uid=1221940  FOL  neg
    └── uid=473616  NON  neg
```

> **Edge tally:** FOL=9  NON=2  UNK=0  →  observed follower ratio: 82%

---

### [4/20] Cascade `9494514` — C2
Size=25 · Depth=4 · Breadth=14 · Follower=0.92 · Sentiment: 72% neg / 4% neu / 24% pos · Event: 0

**Root text:**
> #RamadanKareem to all who observe. May this holy month of fasting, faith, family and
> charity ease the sufferings of the oppressed, everyone’s prayers be answered, blessings
> shared. #RamadanMubarak especially to those in Palestine facing a genocide even during
> the holy month. Lasting #ceasefire now.

**Cascade tree:**
```
ROOT  uid=8906  sent=pos
├── uid=2171  FOL  pos
│   └── uid=167296  FOL  pos
├── uid=878714  FOL  pos
├── uid=2481  FOL  pos
│   ├── uid=1311129  FOL  pos
│   │   └── uid=246243  NON  pos
│   │       └── uid=676712  FOL  pos
│   ├── uid=152797  FOL  pos
│   │   └── uid=30627  FOL  pos
│   ├── uid=1833557  FOL  pos
│   ├── uid=1145804  FOL  pos
│   ├── uid=530963  FOL  pos
│   │   └── uid=331516  FOL  pos
│   ├── uid=1146235  FOL  pos
│   │   └── uid=760942  NON  pos
│   ├── uid=1147892  FOL  pos
│   ├── uid=37696  FOL  pos
│   ├── uid=518413  FOL  pos
│   ├── uid=690243  FOL  pos
│   │   └── uid=169182  FOL  pos
│   ├── uid=162025  FOL  pos
│   ├── uid=203130  FOL  pos
│   └── uid=66686  FOL  pos
└── uid=476281  FOL  pos
```

> **Edge tally:** FOL=22  NON=2  UNK=0  →  observed follower ratio: 92%

---

### [5/20] Cascade `11794490` — C2
Size=10 · Depth=4 · Breadth=4 · Follower=0.78 · Sentiment: 70% neg / 0% neu / 30% pos · Event: 0

**Root text:**
> One thing I want to be clear about:  The occupation of Gaza & the West Bank is a crime
> against humanity that has robbed the dignity of millions & the lives of 10,000s.   I
> don’t think most Americans comprehend the brutal and inhumane treatment Palestinians
> face every day.   Nothing can justify it.

**Cascade tree:**
```
ROOT  uid=410  sent=neg
├── uid=810242  NON  neg
│   ├── uid=61615  NON  neg
│   │   └── uid=33942  FOL  neg
│   │       ├── uid=32274  FOL  neg
│   │       └── uid=381410  FOL  neg
│   └── uid=195198  FOL  neg
├── uid=30041  FOL  neg
├── uid=91084  FOL  neg
└── uid=30092  FOL  neg
```

> **Edge tally:** FOL=7  NON=2  UNK=0  →  observed follower ratio: 78%

---

### [6/20] Cascade `10694813` — C2
Size=7 · Depth=2 · Breadth=4 · Follower=0.83 · Sentiment: 86% neg / 0% neu / 14% pos · Event: 1

**Root text:**
> “During the Holocaust, Nazis referred to Jews as rats. Hutus involved in the Rwanda
> genocide called Tutsis cockroaches. Slave owners throughout history considered slaves
> subhuman animals.” www.npr.org/2011/03/29/1...

**Cascade tree:**
```
ROOT  uid=42583  sent=neg
├── uid=559761  FOL  neg
│   └── uid=82418  FOL  neg
├── uid=1076859  FOL  neg
├── uid=595720  FOL  neg
└── uid=30006  FOL  neg
    └── uid=1824194  NON  neg
```

> **Edge tally:** FOL=5  NON=1  UNK=0  →  observed follower ratio: 83%

---

### [7/20] Cascade `38474958` — C2
Size=6 · Depth=4 · Breadth=2 · Follower=0.80 · Sentiment: 83% neg / 0% neu / 17% pos · Event: 0

**Root text:**
> i love the point you made. everyone i see saying genocide is a dealbreaker is a multiply
> marginalized person. the people telling me in doing so i’m killing democracy are largely
> uh, not.

**Cascade tree:**
```
ROOT  uid=6407  sent=pos
├── uid=854096  FOL  pos
│   └── uid=456837  FOL  pos
│       └── uid=46509  NON  neg  ← quote post
│             | content note: mention of genocide of Jewish ppl  first of all, nothing
│             | in this post is true except for what he admits to his great-grandfather
│             | doing  but for those outside of Europe: this white German's disgusting
│             | views are absolutely common among white Europeans on the left,
│             | especially in Germany.
│           └── uid=282831  FOL  pos
└── uid=98484  FOL  pos
```

> **Edge tally:** FOL=4  NON=1  UNK=0  →  observed follower ratio: 80%

---

### [8/20] Cascade `46072594` — C2
Size=6 · Depth=1 · Breadth=5 · Follower=1.00 · Sentiment: 83% neg / 0% neu / 17% pos · Event: 0

**Root text:**
> Several pro-Palestine graffiti messages were tagged around Center City Philadelphia
> during last night's mass protest demanding an end to mass killings of civilians in Gaza:

**Cascade tree:**
```
ROOT  uid=6758  sent=neg
├── uid=547914  FOL  neg
├── uid=27962  FOL  neg
├── uid=204784  FOL  neg
├── uid=879236  FOL  neg
└── uid=566714  FOL  neg
```

> **Edge tally:** FOL=5  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [9/20] Cascade `15336121` — C2
Size=6 · Depth=2 · Breadth=3 · Follower=0.80 · Sentiment: 83% neg / 0% neu / 17% pos · Event: 0

**Root text:**
> After months of warnings of famine in Gaza, some children begin to succumb in rising
> deaths

**Cascade tree:**
```
ROOT  uid=12285  sent=neg
├── uid=620600  FOL  neg
│   └── uid=64492  NON  neg
├── uid=775981  FOL  neg
│   └── uid=459089  FOL  neg
└── uid=921709  FOL  neg
```

> **Edge tally:** FOL=4  NON=1  UNK=0  →  observed follower ratio: 80%

---

### [10/20] Cascade `5852494` — C2
Size=9 · Depth=4 · Breadth=3 · Follower=0.75 · Sentiment: 89% neg / 0% neu / 11% pos · Event: 0

**Root text:**
> I cried watching the English subtitled version 🥹  We’ve come a long way from Hamas being
> stereotyped as terrorist monsters … to their speeches being subtitled, shared and
> respected as those of freedom fighters.   The blood of our martyrs has not been in vain
> 🥹  Palestine WILL be free !  🇵🇸

**Cascade tree:**
```
ROOT  uid=27939  sent=pos
├── uid=932968  NON  pos
├── uid=27962  FOL  pos
└── uid=9781  FOL  pos
    ├── uid=616277  NON  neu
    ├── uid=161275  FOL  pos
    │   └── uid=596040  FOL  pos
    │       └── uid=200724  FOL  pos
    └── uid=715835  FOL  pos
```

> **Edge tally:** FOL=6  NON=2  UNK=0  →  observed follower ratio: 75%

---

### [11/20] Cascade `18027678` — C3
Size=9 · Depth=2 · Breadth=7 · Follower=0.88 · Sentiment: 67% neg / 11% neu / 22% pos · Event: 2

**Root text:**
> My friend shared this compilation of mental health resources for Palestinian diaspora &
> SWANA people at this time.

**Cascade tree:**
```
ROOT  uid=37717  sent=neu
├── uid=54494  FOL  neu
├── uid=878175  FOL  neu
│   └── uid=530737  NON  neu
├── uid=275910  FOL  neu
├── uid=68880  FOL  neu
├── uid=355566  FOL  neu
├── uid=345458  FOL  neu
└── uid=32421  FOL  neu
```

> **Edge tally:** FOL=7  NON=1  UNK=0  →  observed follower ratio: 88%

---

### [12/20] Cascade `149860` — C3
Size=29 · Depth=4 · Breadth=12 · Follower=0.75 · Sentiment: 62% neg / 31% neu / 7% pos · Event: 0

**Root text:**
> BREAKING  The IDF says that three Israeli hostages were mistakenly identified as threats
> and killed.   The two hostages whose families allowed them to be publicly identified are
> Yotam Haim and Samir Talalka. The family of the third hostage asked that they not be
> identified publicly.

**Cascade tree:**
```
ROOT  uid=7331  sent=neg
├── uid=678931  FOL  neg
│   └── uid=658646  NON  neg
├── uid=383149  FOL  neg
├── uid=161191  FOL  neg
├── uid=1038069  FOL  neg
│   └── uid=619876  NON  neg
│       └── uid=361098  NON  neg
│           └── uid=530925  NON  neg
├── uid=253078  FOL  neg
│   └── uid=34387  NON  neg
│       └── uid=64827  FOL  neg
├── uid=52153  FOL  neg
│   └── uid=46072  FOL  neg
├── uid=297965  FOL  neg
├── uid=1311675  FOL  neg
│   └── uid=311373  NON  neg
│       └── uid=297646  NON  neg
├── uid=181498  FOL  neg
├── uid=32433  FOL  neg
├── uid=6624  FOL  neg
│   ├── uid=3269  FOL  neg
│   │   ├── uid=4121  FOL  neg
│   │   └── uid=726633  FOL  neg
│   ├── uid=66421  FOL  neg
│   ├── uid=217125  FOL  neg
│   │   └── uid=152994  FOL  neg
│   └── uid=65602  FOL  neg
└── uid=713859  FOL  neg
```

> **Edge tally:** FOL=21  NON=7  UNK=0  →  observed follower ratio: 75%

---

### [13/20] Cascade `11165484` — C3
Size=8 · Depth=5 · Breadth=2 · Follower=0.29 · Sentiment: 38% neg / 25% neu / 38% pos · Event: 0

**Root text:**
> I think a lot of us are both feeling the grief/anger of what Israel is doing in Gaza
> (and the West bank), and the grief of watching folks we love entrench themselves - while
> also knowing this is absolutely feeding Christian nationalist led hatred. We have to
> keep speaking out

**Cascade tree:**
```
ROOT  uid=20945  sent=neg
├── uid=366699  NON  neg
│   └── uid=52104  NON  neg
│       ├── uid=30112  NON  neg
│       └── uid=32921  FOL  neg
│           └── uid=726851  NON  neg
│               └── uid=208331  NON  neg
└── uid=150144  FOL  neg
```

> **Edge tally:** FOL=2  NON=5  UNK=0  →  observed follower ratio: 29%

---

### [14/20] Cascade `6542190` — C3
Size=16 · Depth=9 · Breadth=4 · Follower=0.67 · Sentiment: 56% neg / 38% neu / 6% pos · Event: 0

**Root text:**
> Genocide shouldn’t be hard to condemn. It shouldn’t be something anxiety or fear should
> stop you from speaking out against.

**Cascade tree:**
```
ROOT  uid=24933  sent=neg
├── uid=167296  FOL  neg
│   ├── uid=595118  NON  neg
│   │   └── uid=459089  FOL  neg
│   │       └── uid=65926  NON  neg
│   │           └── uid=443390  NON  neg
│   │               └── uid=122771  NON  neg
│   │                   └── uid=716031  NON  neg
│   │                       └── uid=28011  FOL  neg
│   │                           ├── uid=1382246  FOL  neg
│   │                           └── uid=36015  FOL  neg
│   ├── uid=361113  FOL  neg
│   └── uid=46670  FOL  neg
├── uid=570998  FOL  neg
├── uid=433383  FOL  neg
└── uid=26049  FOL  neg
```

> **Edge tally:** FOL=10  NON=5  UNK=0  →  observed follower ratio: 67%

---

### [15/20] Cascade `7016626` — C3
Size=24 · Depth=7 · Breadth=5 · Follower=0.39 · Sentiment: 54% neg / 42% neu / 4% pos · Event: 0

**Root text:**
> Victims of Israel’s war on Gaza

**Cascade tree:**
```
ROOT  uid=46788  sent=neg
├── uid=181592  NON  neg
│   ├── uid=21585  FOL  neg
│   │   └── uid=546909  NON  neg
│   └── uid=99607  NON  neg
│       └── uid=233952  NON  neg
│           └── uid=66817  NON  neg
├── uid=75789  FOL  neg
│   └── uid=33064  NON  neg
│       └── uid=183339  NON  neg
├── uid=90394  FOL  neg
│   └── uid=669592  FOL  neg
│       └── uid=85056  NON  neg
│           └── uid=32809  FOL  neg
│               └── uid=6683  FOL  neg
│                   └── uid=6608  FOL  neg
│                       └── uid=79117  NON  neg
├── uid=164434  FOL  neg
│   └── uid=15130  NON  neg
│       └── uid=361189  NON  neg
│           └── uid=361124  NON  neg
│               └── uid=215375  NON  neg
│                   └── uid=158753  NON  neg
└── uid=88914  FOL  neg
```

> **Edge tally:** FOL=9  NON=14  UNK=0  →  observed follower ratio: 39%

---

### [16/20] Cascade `24351660` — C3
Size=17 · Depth=6 · Breadth=3 · Follower=0.50 · Sentiment: 59% neg / 29% neu / 12% pos · Event: 0

**Root text:**
> Folks, I know nobody is going to like this, but this is literally the result of Blinken
> visiting Israel and playing hardball.  Bibi is now openly contradicting his own
> ministers' earlier statements regarding their genocidal intent.

**Cascade tree:**
```
ROOT  uid=19441  sent=neg
├── uid=703309  NON  neg
│   └── uid=408719  NON  neg
│       └── uid=163314  FOL  neg
│           ├── uid=32342  FOL  neg
│           │   └── uid=423142  NON  neg
│           ├── uid=59143  FOL  neg
│           │   ├── uid=545824  NON  neg
│           │   │   └── uid=1174438  NON  neg
│           │   └── uid=66320  FOL  neg
│           └── uid=65509  FOL  neg
├── uid=2059155  FOL  neg
│   └── uid=65409  NON  neg
│       └── uid=919652  NON  neg
└── uid=158015  FOL  neg
    └── uid=333310  NON  neg
        └── uid=341486  FOL  neg
```

> **Edge tally:** FOL=8  NON=8  UNK=0  →  observed follower ratio: 50%

---

### [17/20] Cascade `882945` — C3
Size=148 · Depth=21 · Breadth=18 · Follower=0.56 · Sentiment: 66% neg / 27% neu / 7% pos · Event: 0

**Root text:**
> Elon doing overtly antisemitic shit and running for cover by censoring posts critical of
> the rightwing government of Israel is a pretty good microcosm of what every rightwing
> antisemite does

**Cascade tree:**
```
ROOT  uid=1249  sent=neg
├── uid=1038146  NON  neg
│   └── uid=71678  NON  neg
│       └── uid=1020535  NON  neg
│           └── uid=288246  NON  neg
│               └── uid=361212  NON  neg
│                   ├── uid=713859  FOL  neg
│                   │   └── uid=37747  NON  neg
│                   │       └── uid=19453  NON  neg
│                   │           ├── uid=592082  FOL  neg
│                   │           └── uid=412923  NON  neg
│                   │               └── uid=533316  NON  neg
│                   │                   └── uid=585583  NON  neg
│                   │                       └── uid=183285  NON  neg
│                   │                           └── uid=1979468  NON  neg
│                   │                               └── uid=417413  NON  neg
│                   │                                   └── uid=2213638  NON  neg
│                   │                                       └── uid=1833935  NON  neg
│                   │                                           └── uid=388790  NON  neg
│                   │                                               └── uid=2059155  NON  neg
│                   │                                                   ├── uid=85779  FOL  neg
│                   │                                                   │   └── uid=880136  FOL  neg
│                   │                                                   └── uid=199505  NON  neg
│                   │                                                       └── uid=268368  NON  neg
│                   └── uid=33026  NON  neg
│                       ├── uid=446529  NON  neg
│                       │   └── uid=236648  FOL  neg
│                       │       └── uid=705128  NON  neg
│                       └── uid=66709  FOL  neg
│                           └── uid=239106  NON  neg
│                               ├── uid=186033  FOL  neg
│                               │   ├── uid=32809  FOL  neg
│                               │   ├── uid=3245145  FOL  neg
│                               │   ├── uid=311231  FOL  neg
│                               │   └── uid=853707  FOL  neg
│                               │       └── uid=381042  NON  neg
│                               │           ├── uid=30622  FOL  neg
│                               │           └── uid=247731  NON  neg
│                               │               └── uid=2817899  NON  neg
│                               │                   └── uid=226627  FOL  neg
│                               │                       ├── uid=2454  FOL  neg
│                               │                       │   ├── uid=524283  FOL  neg
│                               │                       │   ├── uid=27521  FOL  neg
│                               │                       │   │   ├── uid=1018575  NON  neg
│                               │                       │   │   ├── uid=100662  FOL  neg
│                               │                       │   │   └── uid=5750  FOL  neg
│                               │                       │   │       ├── uid=676486  FOL  neg
│                               │                       │   │       ├── uid=19477  FOL  neg
│                               │                       │   │       │   └── uid=332378  FOL  neg
│                               │                       │   │       ├── uid=44111  FOL  neg
│                               │                       │   │       │   └── uid=271292  FOL  neg
│                               │                       │   │       ├── uid=383149  FOL  neg
│                               │                       │   │       │   └── uid=32285  FOL  neg
│                               │                       │   │       ├── uid=440712  FOL  neg
│                               │                       │   │       ├── uid=30081  FOL  neg
│                               │                       │   │       ├── uid=121844  FOL  neg
│                               │                       │   │       └── uid=163789  FOL  neg
│                               │                       │   ├── uid=47357  FOL  neg
│                               │                       │   ├── uid=977685  FOL  neg
│                               │                       │   ├── uid=143283  FOL  neg
```

> **Edge tally:** FOL=82  NON=65  UNK=0  →  observed follower ratio: 56%

---

### [18/20] Cascade `6395008` — C3
Size=33 · Depth=7 · Breadth=18 · Follower=0.75 · Sentiment: 52% neg / 36% neu / 12% pos · Event: 0

**Root text:**
> A reminder Sundays are a great time to contact your representatives and leave a message
> instead of talking to a person. Or you can also find their fax numbers on their website
> and use a free fax website if their message box is full.  5calls.org/issue/israel...

**Cascade tree:**
```
ROOT  uid=236646  sent=pos
└── uid=71678  NON  pos
    └── uid=683609  NON  pos
        └── uid=158293  NON  pos
            ├── uid=123135  FOL  pos
            │   ├── uid=584225  FOL  pos
            │   ├── uid=88649  FOL  pos
            │   ├── uid=1215546  FOL  pos
            │   ├── uid=3060  FOL  pos
            │   │   ├── uid=718751  FOL  pos
            │   │   ├── uid=2056279  FOL  pos
            │   │   ├── uid=1489043  FOL  pos
            │   │   ├── uid=30841  FOL  pos
            │   │   │   ├── uid=171926  FOL  pos
            │   │   │   └── uid=676712  NON  pos
            │   │   ├── uid=67727  FOL  pos
            │   │   │   └── uid=879760  NON  pos
            │   │   ├── uid=1147061  FOL  pos
            │   │   ├── uid=3200  FOL  pos
            │   │   ├── uid=879520  FOL  pos
            │   │   ├── uid=66680  FOL  pos
            │   │   ├── uid=518915  FOL  pos
            │   │   ├── uid=508697  FOL  pos
            │   │   ├── uid=150095  FOL  pos
            │   │   ├── uid=459434  FOL  pos
            │   │   ├── uid=531049  FOL  pos
            │   │   ├── uid=34491  FOL  pos
            │   │   ├── uid=66045  FOL  pos
            │   │   ├── uid=479138  FOL  pos
            │   │   └── uid=3020  FOL  pos
            │   └── uid=32852  NON  pos
            └── uid=199139  NON  pos
                └── uid=30401  NON  pos
```

> **Edge tally:** FOL=24  NON=8  UNK=0  →  observed follower ratio: 75%

---

### [19/20] Cascade `14194985` — C3
Size=16 · Depth=6 · Breadth=4 · Follower=0.60 · Sentiment: 69% neg / 19% neu / 12% pos · Event: 0

**Root text:**
> "[Palestinians] do not believe - nor should they - that they would be allowed to return
> to Gaza...To the extent...a humanitarian corridor accomplishes anything, it would be to
> accelerate the depopulation of Gaza...It would also offers...[a] roadmap for doing the
> same in the West Bank & Jerusalem."

**Cascade tree:**
```
ROOT  uid=32809  sent=neg
├── uid=1874193  NON  neg
│   └── uid=65819  NON  neg
└── uid=6611  FOL  neg
    ├── uid=1396781  NON  neg
    ├── uid=1248  FOL  neg
    │   ├── uid=726597  FOL  neg
    │   │   └── uid=215433  NON  neg
    │   │       └── uid=6683  FOL  neg
    │   │           └── uid=130517  FOL  neg
    │   ├── uid=341573  FOL  neg
    │   │   └── uid=530989  NON  neg
    │   │       ├── uid=966427  NON  neg
    │   │       └── uid=203346  FOL  neg
    │   └── uid=203217  FOL  neg
    └── uid=33064  FOL  neg
```

> **Edge tally:** FOL=9  NON=6  UNK=0  →  observed follower ratio: 60%

---

### [20/20] Cascade `4757999` — C3
Size=12 · Depth=5 · Breadth=5 · Follower=0.82 · Sentiment: 25% neg / 58% neu / 17% pos · Event: 1

**Root text:**
> Lots of convos on Russian, Hezbollah, Iranian involvement. TL;DR - not that simple,
> please don't over-attribute. Some thoughts -   1) Hamas is capable of orchestrating this
> themselves, and they are the principally motivated organization here.

**Cascade tree:**
```
ROOT  uid=8013  sent=neu
├── uid=45666  FOL  neu
│   └── uid=220786  FOL  neu
│       └── uid=85743  FOL  neu
│           └── uid=6677  FOL  neu
│               └── uid=1019233  FOL  neu
├── uid=142883  FOL  neu
│   └── uid=7351  NON  neu
├── uid=710323  FOL  neu
├── uid=566594  FOL  neu
└── uid=439866  FOL  neu
    └── uid=27847  NON  neu
```

> **Edge tally:** FOL=9  NON=2  UNK=0  →  observed follower ratio: 82%

---

## Macro 3 — Follower-driven neutral cascades

**Clusters:** C4, C5  **Total cascades:** 2,509

**Expected profile:**
- Size (median): 8.0  (IQR 7.0–12.0)
- Depth: 3.0  (IQR 2.0–5.0)
- Breadth: 4.0  (IQR 3.0–6.0)
- Follower ratio: 0.9  (IQR 0.8–1.0)
- Sentiment: 0% neg / 100% neu / 0% pos

**What to verify:**
- Predominantly `neu` sentiment
- High `FOL` ratio — tight follower-network propagation
- Moderate size; low-to-moderate depth

---

### [1/20] Cascade `22856574` — C4
Size=6 · Depth=4 · Breadth=2 · Follower=0.80 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 2

**Root text:**
> Für alle, die noch nach Infos zum Angriff der Hamas suchen, um z. B.  im Unterricht
> darüber zu sprechen, die Bundeszentrale für politische Bildung hat hier Materialien
> zusammengestellt. www.bpb.de/kurz-knapp/h...  #twlz #bluelz #bslz

**Cascade tree:**
```
ROOT  uid=280985  sent=neu
└── uid=599589  FOL  neu
    ├── uid=704167  NON  neu
    └── uid=231420  FOL  neu
        └── uid=14336  FOL  neu
            └── uid=279398  FOL  neu
```

> **Edge tally:** FOL=4  NON=1  UNK=0  →  observed follower ratio: 80%

---

### [2/20] Cascade `28898961` — C4
Size=9 · Depth=5 · Breadth=3 · Follower=0.88 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> Qu'espère le Hamas ? "Passer pour des martyrs, asseoir leur drapeau sur la cause
> palestinienne… Ses chefs n’ont aucun problème à regarder les Gazaouis vivre l’enfer
> depuis leurs villas climatisées au Qatar".  www.franc-tireur.fr/pieges-par-d...

**Cascade tree:**
```
ROOT  uid=129731  sent=neu
├── uid=79919  NON  neu
│   ├── uid=80009  FOL  neu
│   ├── uid=79907  FOL  neu
│   └── uid=79953  FOL  neu
│       └── uid=79991  FOL  neu
│           └── uid=79851  FOL  neu
│               └── uid=560007  FOL  neu
└── uid=79963  FOL  neu
```

> **Edge tally:** FOL=7  NON=1  UNK=0  →  observed follower ratio: 88%

---

### [3/20] Cascade `23907413` — C4
Size=8 · Depth=3 · Breadth=4 · Follower=0.71 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> Es ist der US-AMERIKANISCHE Außenminister - of all people - der genau jetzt so klar
> benennt, was passieren muss. Das muss dieser blinde "Gehorsam" gegenüber Israel sein,
> der den Amis auf jeder Demo vorgehalten wird. Sonst wer da, der das heute so
> unmissverständlich einfordert? orf.at/stories/3340...

**Cascade tree:**
```
ROOT  uid=78464  sent=neu
├── uid=106161  NON  neu
├── uid=796770  FOL  neu
│   └── uid=132595  NON  neu
│       ├── uid=106480  FOL  neu
│       └── uid=150491  FOL  neu
├── uid=797053  FOL  neu
└── uid=106324  FOL  neu
```

> **Edge tally:** FOL=5  NON=2  UNK=0  →  observed follower ratio: 71%

---

### [4/20] Cascade `1608261` — C4
Size=9 · Depth=4 · Breadth=5 · Follower=0.62 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> Junge Menschen bewerben plötzlich Al-Qaida-Propaganda & rechtfertigen Gewalt. Der
> Antisemitismus-Experte Christer Mattsson erklärt, was diese Entwicklung mit Greta
> Thunberg und dem Konflikt in Israel zu tun hat. Ein erhellendes Interview von @@71915
> (S+) www.spiegel.de/ausland/anti...

**Cascade tree:**
```
ROOT  uid=29842  sent=neu
├── uid=203802  FOL  neu
│   └── uid=179175  NON  neu
│       └── uid=73586  NON  neu
│           └── uid=57944  NON  neu
├── uid=151810  FOL  neu
├── uid=178541  FOL  neu
├── uid=106324  FOL  neu
└── uid=501923  FOL  neu
```

> **Edge tally:** FOL=5  NON=3  UNK=0  →  observed follower ratio: 62%

---

### [5/20] Cascade `30626223` — C4
Size=10 · Depth=5 · Breadth=4 · Follower=0.89 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> Derzeit erreichen uns mehrere Beschwerden wegen der Veröffentlichung von Foto- und
> Videomaterial zum terroristischen Großangriff der Hamas gegen Israel 🇮🇱. Der Senat 1
> nimmt dies zum Anlass, eine allgemeine Stellungnahme zur Terror- und
> Kriegsberichterstattung abzugeben. (1/5)

**Cascade tree:**
```
ROOT  uid=135701  sent=neu
├── uid=78459  FOL  neu
├── uid=1471175  FOL  neu
│   └── uid=73586  NON  neu
│       └── uid=159887  FOL  neu
│           └── uid=29852  FOL  neu
│               ├── uid=420889  FOL  neu
│               └── uid=178487  FOL  neu
├── uid=130968  FOL  neu
└── uid=127481  FOL  neu
```

> **Edge tally:** FOL=8  NON=1  UNK=0  →  observed follower ratio: 89%

---

### [6/20] Cascade `28029731` — C4
Size=10 · Depth=7 · Breadth=2 · Follower=0.78 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 2

**Root text:**
> Hey Greta Thunberg, if you want a ceasefire, tell this Hamas, not Israel.

**Cascade tree:**
```
ROOT  uid=238322  sent=neu
└── uid=52850  FOL  neu
    ├── uid=318794  NON  neu
    │   └── uid=159519  NON  neu
    │       └── uid=63401  FOL  neu
    │           ├── uid=15290  FOL  neu
    │           │   └── uid=49454  FOL  neu
    │           │       └── uid=14602  FOL  neu
    │           └── uid=147167  FOL  neu
    └── uid=470772  FOL  neu
```

> **Edge tally:** FOL=7  NON=2  UNK=0  →  observed follower ratio: 78%

---

### [7/20] Cascade `18957583` — C4
Size=15 · Depth=8 · Breadth=4 · Follower=0.64 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> Akademischer Antisemitismus: @@4135824 positioniert sich einseitig im #Gazakrieg, kennt
> den Sündenbock und verharmlost #Hamas-Terror. Mein Kommentar. #Leipzig kreuzer-
> leipzig.de/2023/10/28/e...

**Cascade tree:**
```
ROOT  uid=336621  sent=neu
├── uid=152652  NON  neu
│   └── uid=238403  NON  neu
│       └── uid=925914  NON  neu
│           └── uid=132601  NON  neu
│               ├── uid=171581  NON  neu
│               │   └── uid=221343  FOL  neu
│               │       ├── uid=159521  FOL  neu
│               │       └── uid=152372  FOL  neu
│               │           └── uid=14578  FOL  neu
│               └── uid=12913  FOL  neu
│                   └── uid=120562  FOL  neu
├── uid=1381359  FOL  neu
├── uid=71984  FOL  neu
└── uid=4542  FOL  neu
```

> **Edge tally:** FOL=9  NON=5  UNK=0  →  observed follower ratio: 64%

---

### [8/20] Cascade `40688963` — C4
Size=7 · Depth=5 · Breadth=2 · Follower=0.67 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> Flyktingförläggning i Gaza har ”drabbats av en explosion”. Orkar inte. Har de även
> drabbats av strömavbrott? Matbrist? Isolering? Har de drabbats av döden? Haft lite otur?

**Cascade tree:**
```
ROOT  uid=142976  sent=neu
└── uid=306673  FOL  neu
    └── uid=142820  NON  neu
        └── uid=143006  NON  neu
            └── uid=142773  FOL  neu
                ├── uid=19612  FOL  neu
                └── uid=100317  FOL  neu
```

> **Edge tally:** FOL=4  NON=2  UNK=0  →  observed follower ratio: 67%

---

### [9/20] Cascade `2801955` — C4
Size=6 · Depth=4 · Breadth=2 · Follower=0.60 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> Dass die drei Ampelparteien zusammen mit der Union einen  Antrag zu Israel in den
> Bundestag einbringen werden, und die Linkspartei außen vor lassen, ist nicht
> nachvollziehbar. Dadurch stellen sie diese auf eine Stufe mit AfD und verpassen ein
> wichtiges gemeinsames Zeichen der demokratischen Parteien

**Cascade tree:**
```
ROOT  uid=159517  sent=neu
├── uid=49715  NON  neu
└── uid=443133  FOL  neu
    └── uid=405754  NON  neu
        └── uid=14589  FOL  neu
            └── uid=49454  FOL  neu
```

> **Edge tally:** FOL=3  NON=2  UNK=0  →  observed follower ratio: 60%

---

### [10/20] Cascade `17770195` — C4
Size=8 · Depth=7 · Breadth=1 · Follower=0.43 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 1

**Root text:**
> In Mannheim haben ca. 700 Menschen an einer Mahnwache für #Israel teilgenommen 😔 es kam
> nicht zu Ausschreitungen glücklicherweise

**Cascade tree:**
```
ROOT  uid=581842  sent=neu
└── uid=549429  NON  neu
    └── uid=106603  NON  neu
        └── uid=106094  NON  neu
            └── uid=14821  FOL  neu
                └── uid=13560  NON  neu
                    └── uid=3379  FOL  neu
                        └── uid=136798  FOL  neu
```

> **Edge tally:** FOL=3  NON=4  UNK=0  →  observed follower ratio: 43%

---

### [11/20] Cascade `28577240` — C5
Size=14 · Depth=3 · Breadth=10 · Follower=1.00 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 1

**Root text:**
> "Im Hinblick auf den terroristischen Angriff auf Israel und die Sicherheitslage für jüd.
> Einrichtungen in Deutschland stehen wir (Zentralrat der Juden in Deutschland) im
> intensiven Austausch mit den Sicherheitsbehörden. Die polizeilichen Maßnahmen vor jüd.
> Einrichtungen werden bundesweit erhöht."

**Cascade tree:**
```
ROOT  uid=51759  sent=neu
├── uid=720870  FOL  neu
├── uid=152481  FOL  neu
│   └── uid=151608  FOL  neu
├── uid=89258  FOL  neu
├── uid=49718  FOL  neu
│   └── uid=51825  FOL  neu
│       └── uid=14475  FOL  neu
├── uid=258576  FOL  neu
├── uid=581535  FOL  neu
├── uid=432380  FOL  neu
├── uid=13938  FOL  neu
├── uid=1556303  FOL  neu
└── uid=152372  FOL  neu
```

> **Edge tally:** FOL=13  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [12/20] Cascade `6005189` — C5
Size=7 · Depth=3 · Breadth=4 · Follower=1.00 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
>  #StopGenocide  #StopKillingChildren  #FreeGaza  #FreePalestine

**Cascade tree:**
```
ROOT  uid=631869  sent=neu
├── uid=1110456  FOL  neu
├── uid=464117  FOL  neu
│   └── uid=230698  FOL  neu
│       └── uid=860891  FOL  neu
├── uid=230727  FOL  neu
└── uid=261811  FOL  neu
```

> **Edge tally:** FOL=6  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [13/20] Cascade `5149811` — C5
Size=17 · Depth=2 · Breadth=12 · Follower=1.00 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 3

**Root text:**
> Raketenfehlschuss wohl Ursache für Explosion an Gaza-Klinik

**Cascade tree:**
```
ROOT  uid=1197  sent=neu
├── uid=714520  FOL  neu
├── uid=150496  FOL  neu
├── uid=358511  FOL  neu
├── uid=178410  FOL  neu
├── uid=769272  FOL  neu
├── uid=106716  FOL  neu
│   ├── uid=167963  FOL  neu
│   ├── uid=152399  FOL  neu
│   ├── uid=86956  FOL  neu
│   └── uid=227925  FOL  neu
├── uid=96382  FOL  neu
├── uid=150894  FOL  neu
├── uid=718009  FOL  neu
├── uid=150406  FOL  neu
├── uid=97815  FOL  neu
└── uid=98847  FOL  neu
```

> **Edge tally:** FOL=16  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [14/20] Cascade `28648826` — C5
Size=7 · Depth=2 · Breadth=4 · Follower=0.83 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> "Auf Instagram sprechen Aktivisten um Greta Thunberg von Israel als einem
> "Apartheidssystem" – und von "Lügen" westlicher Medien."  "Die deutsche Sektion
> distanziert sich."  www.zeit.de/politik/ausl...

**Cascade tree:**
```
ROOT  uid=15195  sent=neu
├── uid=12912  FOL  neu
├── uid=238435  FOL  neu
├── uid=12877  FOL  neu
│   └── uid=152652  NON  neu
└── uid=17802  FOL  neu
    └── uid=14756  FOL  neu
```

> **Edge tally:** FOL=5  NON=1  UNK=0  →  observed follower ratio: 83%

---

### [15/20] Cascade `36983658` — C5
Size=9 · Depth=2 · Breadth=7 · Follower=1.00 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 2

**Root text:**
> Ziel muss auch sein, den Palästinensern zu vermitteln, dass sich der Plan der Hamas auch
> gegen sie und die ganze Region richtet. Sie sind ebenfalls Geiseln dieser Terrorgruppe,
> der es nur um sich, die Interessen des Irans und der Hisbollah geht.

**Cascade tree:**
```
ROOT  uid=9509  sent=neu
├── uid=181074  FOL  neu
├── uid=2304216  FOL  neu
├── uid=518598  FOL  neu
├── uid=73610  FOL  neu
├── uid=52850  FOL  neu
│   └── uid=17749  FOL  neu
├── uid=437486  FOL  neu
└── uid=106626  FOL  neu
```

> **Edge tally:** FOL=8  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [16/20] Cascade `20829860` — C5
Size=7 · Depth=3 · Breadth=4 · Follower=1.00 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> Israel ya está vendiendo las concesiones sobre el territorio que van a desalojar
> www.reuters.com/business/ene...

**Cascade tree:**
```
ROOT  uid=10546  sent=neu
├── uid=677564  FOL  neu
├── uid=52498  FOL  neu
│   └── uid=644396  FOL  neu
│       └── uid=683671  FOL  neu
├── uid=322320  FOL  neu
└── uid=406124  FOL  neu
```

> **Edge tally:** FOL=6  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [17/20] Cascade `9023001` — C5
Size=11 · Depth=2 · Breadth=6 · Follower=0.70 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> Weil Jüdinnen und Juden angesichts des Terrors der Hamas wieder um ihre Sicherheit
> fürchten müssen und auch in Deutschland wieder antisemitische Parolen skandiert werden,
> debattieren die Abgeordneten des Bundestages heute ab 9 Uhr zum Schutz des jüdischen
> Lebens in Deutschland. Zum Livestream: (2/2)

**Cascade tree:**
```
ROOT  uid=1536  sent=neu
├── uid=326231  FOL  neu
│   └── uid=337126  NON  neu
├── uid=63238  FOL  neu
│   └── uid=9387  FOL  neu
├── uid=1806282  FOL  neu
├── uid=98720  FOL  neu
│   └── uid=48393  NON  neu
├── uid=220488  FOL  neu
└── uid=106452  FOL  neu
    └── uid=14756  NON  neu
```

> **Edge tally:** FOL=7  NON=3  UNK=0  →  observed follower ratio: 70%

---

### [18/20] Cascade `19712089` — C5
Size=10 · Depth=3 · Breadth=4 · Follower=0.67 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> @@364217 (RLS Israel) in analyse&kritik über die Isolation der Politik in Israel, einen
> sich entmenschlichenden Diskurs gegenüber Palästinenser*innen und Kriegsgegner*innen und
> die zunehmende Repression gegen Antikriegsdemonstrationen.

**Cascade tree:**
```
ROOT  uid=4395  sent=neu
├── uid=506879  NON  neu
│   └── uid=292831  FOL  neu
│       └── uid=658646  NON  neu
├── uid=1686  FOL  neu
│   └── uid=4496  NON  neu
└── uid=20197  FOL  neu
    ├── uid=1080350  FOL  neu
    └── uid=153764  FOL  neu
        └── uid=153217  FOL  neu
```

> **Edge tally:** FOL=6  NON=3  UNK=0  →  observed follower ratio: 67%

---

### [19/20] Cascade `3897965` — C5
Size=8 · Depth=2 · Breadth=5 · Follower=0.86 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> Und an alle, die das Leid in Israel und Palästina missbrauchen für ihre politische
> Agenda: ihr widert mich so an.

**Cascade tree:**
```
ROOT  uid=8071  sent=neu
├── uid=581607  FOL  neu
├── uid=83195  FOL  neu
│   └── uid=106203  FOL  neu
├── uid=13152  FOL  neu
├── uid=166324  FOL  neu
└── uid=152468  FOL  neu
    └── uid=153554  NON  neu
```

> **Edge tally:** FOL=6  NON=1  UNK=0  →  observed follower ratio: 86%

---

### [20/20] Cascade `34701808` — C5
Size=6 · Depth=3 · Breadth=2 · Follower=1.00 · Sentiment: 0% neg / 100% neu / 0% pos · Event: 0

**Root text:**
> „Seit dem 7. Oktober 2023 gibt es ein weiteres ,Davor‘ und ,Danach´.“ Folgen des
> Terroranschlags der Hamas auf Überlebende des Halle-Anschlags und die Bedeutung von
> Handlungsfähigkeit"  www.mobile-opferberatung.de/seit-dem-7-o...

**Cascade tree:**
```
ROOT  uid=49723  sent=neu
├── uid=153223  FOL  neu
│   ├── uid=51802  FOL  neu
│   │   └── uid=136155  FOL  neu
│   └── uid=95632  FOL  neu
└── uid=1578016  FOL  neu
```

> **Edge tally:** FOL=5  NON=0  UNK=0  →  observed follower ratio: 100%

---

## Macro 4 — Neutral follower spreading

**Clusters:** C6  **Total cascades:** 3,554

**Expected profile:**
- Size (median): 13.0  (IQR 8.0–26.0)
- Depth: 4.0  (IQR 3.0–6.0)
- Breadth: 6.0  (IQR 4.0–10.0)
- Follower ratio: 0.9  (IQR 0.8–1.0)
- Sentiment: 10% neg / 90% neu / 0% pos

**What to verify:**
- Predominantly `neu` sentiment
- High `FOL` ratio; small-to-moderate cascade size
- Compact, follower-driven spreading pattern

---

### [1/20] Cascade `938472` — C6
Size=22 · Depth=4 · Breadth=10 · Follower=0.86 · Sentiment: 9% neg / 91% neu / 0% pos · Event: 1

**Root text:**
> Die Bilder der brutalen Gewalt der Hamas gegen Zivilisten sind einfach nur schrecklich.
> Es tut mir so leid für all diejenigen, die in diesen Stunden um die Sicherheit von
> Freunden & Angehörigen bangen.

**Cascade tree:**
```
ROOT  uid=8063  sent=neu
├── uid=151605  NON  neu
│   └── uid=737397  FOL  neu
│       └── uid=159515  NON  neu
│           └── uid=106413  NON  neu
├── uid=142261  FOL  neu
│   └── uid=1682221  FOL  neu
├── uid=398326  FOL  neu
├── uid=336664  FOL  neu
├── uid=890451  FOL  neu
├── uid=1688554  FOL  neu
├── uid=49718  FOL  neu
│   ├── uid=581455  FOL  neu
│   ├── uid=120578  FOL  neu
│   │   └── uid=13938  FOL  neu
│   │       ├── uid=13471  FOL  neu
│   │       └── uid=220819  FOL  neu
│   └── uid=229747  FOL  neu
├── uid=13971  FOL  neu
├── uid=3841  FOL  neu
│   └── uid=179152  FOL  neu
└── uid=429415  FOL  neu
```

> **Edge tally:** FOL=18  NON=3  UNK=0  →  observed follower ratio: 86%

---

### [2/20] Cascade `18333382` — C6
Size=14 · Depth=3 · Breadth=9 · Follower=1.00 · Sentiment: 14% neg / 86% neu / 0% pos · Event: 2

**Root text:**
> Obscénité de la gauche-Hamas, des MélenchonLFI et associations niaises/manipulées qui
> ont bloqué l’expulsion de la famille de l’assassin Djihadiste.  Obscénité de la droite-
> AssadPoutine, des LePenZemmour qui osent “on vous l’avait dit”, eux les disciples des
> collabos antisémites.

**Cascade tree:**
```
ROOT  uid=141371  sent=neu
├── uid=845398  FOL  neu
├── uid=80074  FOL  neu
│   ├── uid=79972  FOL  neu
│   └── uid=845390  FOL  neu
├── uid=1200228  FOL  neu
├── uid=141469  FOL  neu
│   └── uid=79990  FOL  neu
│       └── uid=79991  FOL  neu
├── uid=551297  FOL  neu
├── uid=449542  FOL  neu
├── uid=296149  FOL  neu
├── uid=174631  FOL  neu
└── uid=174061  FOL  neu
```

> **Edge tally:** FOL=13  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [3/20] Cascade `20719427` — C6
Size=32 · Depth=6 · Breadth=10 · Follower=0.87 · Sentiment: 3% neg / 97% neu / 0% pos · Event: 3

**Root text:**
> Monitoringbericht: Antisemitische Reaktionen auf den 07. Oktober - Antisemitische
> Vorfälle in #Deutschland im Kontext der Massaker und des Krieges in Israel und Gaza
> zwischen dem 07. Oktober und 09. November 2023. #Antisemitismus  Zum ganzen Bericht:
> www.report-antisemitism.de/documents/20...

**Cascade tree:**
```
ROOT  uid=51759  sent=neu
├── uid=227985  NON  neu
├── uid=106534  FOL  neu
├── uid=40850  FOL  neu
│   ├── uid=4579  FOL  neu
│   │   ├── uid=74434  FOL  neu
│   │   │   └── uid=14602  FOL  neu
│   │   ├── uid=1556303  FOL  neu
│   │   ├── uid=179148  FOL  neu
│   │   ├── uid=130314  FOL  neu
│   │   ├── uid=796170  FOL  neu
│   │   ├── uid=1625  FOL  neu
│   │   │   ├── uid=136155  FOL  neu
│   │   │   │   └── uid=14595  FOL  neu
│   │   │   │       └── uid=15175  FOL  neu
│   │   │   ├── uid=953173  FOL  neu
│   │   │   └── uid=152678  FOL  neu
│   │   ├── uid=221343  FOL  neu
│   │   │   └── uid=150467  FOL  neu
│   │   └── uid=152750  FOL  neu
│   └── uid=95641  NON  neu
├── uid=106505  FOL  neu
│   └── uid=304392  FOL  neu
│       └── uid=229450  FOL  neu
│           └── uid=221331  FOL  neu
│               └── uid=13107  FOL  neu
├── uid=4439  FOL  neu
│   └── uid=180550  NON  neu
│       └── uid=152716  FOL  neu
│           └── uid=152342  FOL  neu
│               └── uid=49669  NON  neu
└── uid=227956  FOL  neu
```

> **Edge tally:** FOL=27  NON=4  UNK=0  →  observed follower ratio: 87%

---

### [4/20] Cascade `84823278` — C6
Size=6 · Depth=2 · Breadth=3 · Follower=0.60 · Sentiment: 17% neg / 83% neu / 0% pos · Event: 0

**Root text:**
> aus der jüdisch-bolschewistischen weltverschwörung haben deutsche mittlerweile eine
> linke antisemitische gemacht in der human rights, amnesty, icrc, ärzte ohne grenzen, die
> un, selbsthassenden jüd:innen von btselem uvm alle lügen über israel erzählen und den
> staat delegitimieren und dämonisieren

**Cascade tree:**
```
ROOT  uid=86302  sent=neu
├── uid=17784  NON  neu
│   ├── uid=156346  NON  neu
│   ├── uid=25549  FOL  neu
│   └── uid=17783  FOL  neu
└── uid=527238  FOL  neu
```

> **Edge tally:** FOL=3  NON=2  UNK=0  →  observed follower ratio: 60%

---

### [5/20] Cascade `27059233` — C6
Size=24 · Depth=6 · Breadth=7 · Follower=0.91 · Sentiment: 17% neg / 83% neu / 0% pos · Event: 1

**Root text:**
> My 1st bluesky subtweet. My social media have many decolonial colleagues celebrating
> Hamas violence. They seem undisturbed by hostages, sexual violence, female corpses
> undressed, spat on. That doesn't mean that decolonization has failed; just that some
> people fail at it - when it comes to Jews.

**Cascade tree:**
```
ROOT  uid=127735  sent=neg
├── uid=106234  FOL  neg
│   └── uid=227985  NON  neg
│       └── uid=1747422  FOL  neg
├── uid=73643  FOL  neg
├── uid=1076282  FOL  neg
├── uid=405521  FOL  neg
│   └── uid=637122  FOL  neg
│       └── uid=127481  FOL  neg
├── uid=399609  FOL  neg
├── uid=106684  FOL  neu
└── uid=155123  FOL  neg
    ├── uid=1228  FOL  neg
    │   └── uid=1361798  FOL  neg
    └── uid=1381361  NON  neg
        └── uid=4550  FOL  neg
            ├── uid=475460  FOL  neg
            ├── uid=340362  FOL  neg
            │   └── uid=4587  FOL  neg
            │       └── uid=327217  FOL  neg
            └── uid=4593  FOL  neg
                ├── uid=1206094  FOL  neg
                ├── uid=491918  FOL  neg
                └── uid=29686  FOL  neg
```

> **Edge tally:** FOL=21  NON=2  UNK=0  →  observed follower ratio: 91%

---

### [6/20] Cascade `26331544` — C6
Size=8 · Depth=2 · Breadth=6 · Follower=0.71 · Sentiment: 12% neg / 88% neu / 0% pos · Event: 0

**Root text:**
> "Noch nie so etwas in Gaza gesehen"

**Cascade tree:**
```
ROOT  uid=1197  sent=neu
├── uid=527400  NON  neu
├── uid=234473  FOL  neu
├── uid=1768519  FOL  neu
├── uid=238319  FOL  neu
├── uid=341155  FOL  neu
│   └── uid=1808784  NON  neu
└── uid=71970  FOL  neu
```

> **Edge tally:** FOL=5  NON=2  UNK=0  →  observed follower ratio: 71%

---

### [7/20] Cascade `15486211` — C6
Size=10 · Depth=3 · Breadth=5 · Follower=1.00 · Sentiment: 10% neg / 90% neu / 0% pos · Event: 1

**Root text:**
> Sebastian Kurz gründet in Israel eine Sicherheitsfirma.  Jetzt ist Israel das
> unsicherste Land der Welt.  Zufälle gibt's

**Cascade tree:**
```
ROOT  uid=14994  sent=neu
├── uid=159940  FOL  neu
│   └── uid=280842  FOL  neu
├── uid=159889  FOL  neu
│   └── uid=67165  FOL  neu
│       ├── uid=106397  FOL  neu
│       └── uid=304332  FOL  neu
├── uid=625430  FOL  neu
├── uid=106563  FOL  neu
└── uid=99226  FOL  neu
```

> **Edge tally:** FOL=9  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [8/20] Cascade `7895037` — C6
Size=18 · Depth=2 · Breadth=14 · Follower=0.94 · Sentiment: 6% neg / 94% neu / 0% pos · Event: 0

**Root text:**
> Das Leid in Gaza und Israel ist schwer in Worte zu fassen. Und weil es so schwer in
> Worte zu fassen ist, sollte man nicht in jeder Aussage die bösesten Absichten vermuten.
> Es braucht mehr Raum für Schmerz und Angst und Empathie auch abseits der großen
> politischen Fragen.

**Cascade tree:**
```
ROOT  uid=1186  sent=neu
├── uid=968862  FOL  neu
├── uid=510412  FOL  neu
├── uid=8051  FOL  neu
├── uid=942500  FOL  neu
├── uid=274118  FOL  neu
├── uid=1277276  FOL  neu
├── uid=127726  FOL  neu
├── uid=46805  FOL  neu
├── uid=153610  FOL  neu
├── uid=127732  FOL  neu
│   └── uid=532653  FOL  neu
├── uid=326245  FOL  neu
│   └── uid=96715  FOL  neu
├── uid=1719534  FOL  neu
│   └── uid=276101  NON  neu
├── uid=221440  FOL  neu
└── uid=179102  FOL  neu
```

> **Edge tally:** FOL=16  NON=1  UNK=0  →  observed follower ratio: 94%

---

### [9/20] Cascade `7901631` — C6
Size=86 · Depth=6 · Breadth=45 · Follower=0.85 · Sentiment: 1% neg / 99% neu / 0% pos · Event: 0

**Root text:**
> Hamas, in einem Bild erklärt.

**Cascade tree:**
```
ROOT  uid=3837  sent=neu
├── uid=2877348  NON  neu
├── uid=229668  FOL  neu
├── uid=780933  FOL  neu
├── uid=15181  FOL  neu
│   └── uid=14983  FOL  neu
│       ├── uid=205664  FOL  neu
│       ├── uid=48444  FOL  neu
│       └── uid=98721  FOL  neu
├── uid=167507  FOL  neu
├── uid=717830  FOL  neu
├── uid=96614  FOL  neu
├── uid=1727626  FOL  neu
├── uid=94935  FOL  neu
├── uid=106070  FOL  neu
├── uid=83203  FOL  neu
├── uid=1719102  FOL  neu
├── uid=98863  FOL  neu
│   └── uid=106324  NON  neu
│       └── uid=13471  FOL  neu
│           └── uid=15175  FOL  neu
│               └── uid=17837  FOL  neu
├── uid=304326  FOL  neu
├── uid=151339  FOL  neu
├── uid=41042  FOL  neu
├── uid=178465  FOL  neu
├── uid=1707994  FOL  neu
├── uid=1258654  FOL  neu
├── uid=221440  FOL  neu
│   └── uid=3851  FOL  neu
│       └── uid=1038713  FOL  neu
├── uid=2849464  FOL  neu
├── uid=1719301  FOL  neu
│   └── uid=796704  NON  neu
├── uid=17802  FOL  neu
│   ├── uid=827801  FOL  neu
│   └── uid=83033  FOL  neu
├── uid=157433  FOL  neu
│   ├── uid=12893  FOL  neu
│   └── uid=150400  NON  neu
├── uid=745078  FOL  neu
├── uid=96774  FOL  neu
├── uid=106192  FOL  neu
├── uid=120568  FOL  neu
│   └── uid=2737181  NON  neu
├── uid=1203424  FOL  neu
├── uid=151300  FOL  neu
│   └── uid=688668  NON  neu
├── uid=557368  FOL  neu
├── uid=583162  FOL  neu
│   ├── uid=98882  NON  neu
│   │   └── uid=96507  FOL  neu
│   └── uid=15292  FOL  neu
│       └── uid=691625  FOL  neu
├── uid=355434  FOL  neu
│   └── uid=1661  FOL  neu
│       └── uid=13142  FOL  neu
├── uid=1806652  FOL  neu
├── uid=127831  FOL  neu
│   ├── uid=942438  FOL  neu
```

> **Edge tally:** FOL=72  NON=13  UNK=0  →  observed follower ratio: 85%

---

### [10/20] Cascade `18111830` — C6
Size=42 · Depth=9 · Breadth=18 · Follower=0.80 · Sentiment: 2% neg / 98% neu / 0% pos · Event: 0

**Root text:**
> Differenzierte Argumentation von Navid Kermani, weshalb die Israel-Kritik seit dem
> 7. Oktober so falsch klingt, selbst wo sie richtig ist. „Beinah die einzigen Stimmen,
> die in den vergangenen vier Wochen überzeugend Kritik an Israel übten, waren jüdische.“

**Cascade tree:**
```
ROOT  uid=29691  sent=neu
  | Kermani: „Erst die eindeutige und auch mitfühlende Verurteilung des Hamas-Terrors
  | und die Solidarität mit den Juden, die sich nirgends mehr sicher fühlen können,
  | würde die eingeforderte Differenzierung und die Kritik am israelischen
  | Regierungshandeln wieder möglich machen.“
├── uid=3012687  FOL  neu
├── uid=106239  FOL  neu
│   └── uid=13362  FOL  neu
├── uid=797202  FOL  neu
├── uid=797332  FOL  neu
├── uid=252187  FOL  neu
│   └── uid=527499  NON  neu
├── uid=153612  FOL  neu
├── uid=540096  FOL  neu
├── uid=1806910  FOL  neu
├── uid=581362  FOL  neu
│   └── uid=3379  FOL  neu
├── uid=803902  FOL  neu
├── uid=1741067  FOL  neu
├── uid=1378417  FOL  neu
│   └── uid=744253  NON  neu
├── uid=678258  FOL  neu
│   └── uid=950608  NON  neu
│       └── uid=49673  NON  neu
│           ├── uid=43467  NON  neu
│           │   ├── uid=73641  FOL  neu
│           │   │   ├── uid=14575  FOL  neu
│           │   │   ├── uid=159519  FOL  neu
│           │   │   └── uid=98720  FOL  neu
│           │   └── uid=911331  FOL  neu
│           └── uid=75086  FOL  neu
│               ├── uid=1345829  FOL  neu
│               ├── uid=1228  NON  neu
│               └── uid=71929  FOL  neu
│                   ├── uid=127528  FOL  neu
│                   │   └── uid=11288  NON  neu
│                   │       └── uid=4578  FOL  neu
│                   │           ├── uid=159492  FOL  neu
│                   │           ├── uid=143832  FOL  neu
│                   │           └── uid=179102  FOL  neu
│                   └── uid=98266  NON  neu
├── uid=499299  FOL  neu
├── uid=1719072  FOL  neu
├── uid=562998  FOL  neu
├── uid=518525  FOL  neu
└── uid=826410  FOL  neu
```

> **Edge tally:** FOL=33  NON=8  UNK=0  →  observed follower ratio: 80%

---

### [11/20] Cascade `49345502` — C6
Size=11 · Depth=2 · Breadth=7 · Follower=0.90 · Sentiment: 9% neg / 91% neu / 0% pos · Event: 1

**Root text:**
> Ein guter Kommentar zu Israel von Thomas L. Friedman in der NYT, auch zu den Folgen für
> die Ukraine.  "Ich beobachte, wie das Hamas-Israel-Erdbeben ein weiteres Erdbeben
> auslösen wird."  www.nytimes.com/2023/10/07/o...

**Cascade tree:**
```
ROOT  uid=49409  sent=neu
├── uid=397395  NON  neu
├── uid=96403  FOL  neu
│   ├── uid=638514  FOL  neu
│   ├── uid=74449  FOL  neu
│   └── uid=127708  FOL  neu
├── uid=853946  FOL  neu
├── uid=914422  FOL  neu
├── uid=73634  FOL  neu
├── uid=364216  FOL  neu
└── uid=179102  FOL  neu
```

> **Edge tally:** FOL=9  NON=1  UNK=0  →  observed follower ratio: 90%

---

### [12/20] Cascade `3218641` — C6
Size=29 · Depth=9 · Breadth=9 · Follower=0.86 · Sentiment: 7% neg / 93% neu / 0% pos · Event: 0

**Root text:**
> Innerhalb von drei Wochen verbietet Nancy Faeser die Hamas und die PFLP
> Vorfeldorganisation Samidoun. Ein längst überfälliger, wenn auch wichtiger Schritt in
> der Bekämpfung von Antisemitismus und Terrorverherrlichung in Deutschland. Aktivitäten
> für beide Organisationen sind nun strafbar.

**Cascade tree:**
```
ROOT  uid=1518  sent=neu
├── uid=1889070  FOL  neu
│   └── uid=13971  FOL  neu
│       ├── uid=581571  FOL  neu
│       └── uid=106437  FOL  neu
│           └── uid=150700  FOL  neu
│               └── uid=13576  FOL  neu
│                   ├── uid=17826  FOL  neu
│                   │   └── uid=96842  FOL  neu
│                   │       ├── uid=3379  FOL  neu
│                   │       │   └── uid=17837  FOL  neu
│                   │       ├── uid=2931  FOL  neu
│                   │       └── uid=91742  NON  neu
│                   └── uid=49714  FOL  neu
├── uid=15078  FOL  neu
├── uid=707028  FOL  neu
├── uid=426101  FOL  neu
├── uid=1805497  FOL  neu
│   └── uid=718228  NON  neu
├── uid=14630  FOL  neu
│   └── uid=14594  FOL  neu
│       └── uid=14813  FOL  neu
├── uid=361064  FOL  neu
├── uid=616150  FOL  neu
│   └── uid=106265  NON  neu
│       └── uid=14770  FOL  neu
│           └── uid=221166  NON  neu
│               └── uid=662948  FOL  neu
└── uid=179102  FOL  neu
```

> **Edge tally:** FOL=24  NON=4  UNK=0  →  observed follower ratio: 86%

---

### [13/20] Cascade `11475749` — C6
Size=6 · Depth=2 · Breadth=4 · Follower=0.60 · Sentiment: 17% neg / 83% neu / 0% pos · Event: 0

**Root text:**
> Med den høyreekstreme og rasistiske regjeringen til Benjamin Netanyahu fikk de voldelige
> bosetterne etter nyttår grønt lys til å gjøre omtrent hva de ville med palestinerne.   I
> skyggen av Gaza-krigen går israelske soldater og jødiske bosettere brutalt fram mot
> palestinerne på Vestbredden.

**Cascade tree:**
```
ROOT  uid=60120  sent=neu
├── uid=60033  NON  neu
├── uid=260628  FOL  neu
├── uid=260633  FOL  neu
│   └── uid=260634  NON  neu
└── uid=260699  FOL  neu
```

> **Edge tally:** FOL=3  NON=2  UNK=0  →  observed follower ratio: 60%

---

### [14/20] Cascade `4542059` — C6
Size=12 · Depth=3 · Breadth=6 · Follower=0.91 · Sentiment: 8% neg / 92% neu / 0% pos · Event: 2

**Root text:**
> Hanno Hauenstein @@39038  mit wichtigen Worten zum Umgang der gesellschaftlichen Linken
> mit dem Massaker der Hamas und dem israelischen Angriff auf Gaza.

**Cascade tree:**
```
ROOT  uid=20214  sent=neu
├── uid=98613  FOL  neu
├── uid=86302  FOL  neu
│   └── uid=152600  FOL  neu
│       └── uid=1121923  NON  neu
├── uid=67165  FOL  neu
│   └── uid=83164  FOL  neu
├── uid=4546  FOL  neu
├── uid=14589  FOL  neu
│   └── uid=39038  FOL  neu
│       └── uid=1654967  FOL  neu
└── uid=397923  FOL  neu
```

> **Edge tally:** FOL=10  NON=1  UNK=0  →  observed follower ratio: 91%

---

### [15/20] Cascade `10135273` — C6
Size=7 · Depth=2 · Breadth=4 · Follower=1.00 · Sentiment: 14% neg / 86% neu / 0% pos · Event: 3

**Root text:**
> A ver si Israel tiene 🥚🥚🥚de llamar terrorista a Biden, llamar a regañar al embajador
> estadounidense y decir que USA es un aliado de Hamás.

**Cascade tree:**
```
ROOT  uid=47876  sent=neu
├── uid=656229  FOL  neu
│   └── uid=810291  FOL  neu
├── uid=677594  FOL  neu
├── uid=521048  FOL  neu
│   └── uid=329277  FOL  neu
└── uid=451172  FOL  neu
```

> **Edge tally:** FOL=6  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [16/20] Cascade `4295091` — C6
Size=42 · Depth=7 · Breadth=16 · Follower=0.83 · Sentiment: 2% neg / 95% neu / 2% pos · Event: 0

**Root text:**
> 9. Wie hilft es der palästinensischen Zivilbevölkerung, wenn Hamas israelische Babys
> verbrennt, tötet oder entführt?  10. Sollte es statt „Free Palestine“ nicht „Free
> Palestine from Hamas“ heißen?  Danke für konstruktive Antworten im Voraus. Ach ja: Und
> #BringThemBackHome   (4/4)

**Cascade tree:**
```
ROOT  uid=8214  sent=neu
├── uid=14534  NON  neu
│   ├── uid=17770  NON  neu
│   │   └── uid=152342  NON  neu
│   └── uid=15353  FOL  neu
│       ├── uid=95589  FOL  neu
│       │   └── uid=331030  FOL  neu
│       ├── uid=434886  FOL  neu
│       ├── uid=15001  FOL  neu
│       │   ├── uid=217302  NON  neu
│       │   └── uid=15306  FOL  neu
│       │       ├── uid=13885  FOL  neu
│       │       │   └── uid=16129  NON  neu
│       │       ├── uid=91742  FOL  neu
│       │       ├── uid=518490  NON  neu
│       │       ├── uid=13786  FOL  neu
│       │       ├── uid=280815  FOL  neu
│       │       │   ├── uid=12913  FOL  neu
│       │       │   │   └── uid=17837  FOL  neu
│       │       │   └── uid=823573  FOL  neu
│       │       └── uid=157366  FOL  neu
│       └── uid=293060  FOL  neu
├── uid=205668  FOL  neu
├── uid=1411035  FOL  neu
├── uid=150772  FOL  neu
├── uid=97815  FOL  neu
├── uid=14496  FOL  neu
│   └── uid=440128  FOL  neu
├── uid=98768  FOL  neu
├── uid=423847  FOL  neu
├── uid=98863  FOL  neu
├── uid=843598  FOL  neu
│   └── uid=98753  FOL  neu
├── uid=691572  FOL  neu
├── uid=1427329  FOL  neu
│   └── uid=582329  NON  neu
├── uid=1029278  FOL  neu
├── uid=495462  FOL  neu
├── uid=648088  FOL  neu
│   ├── uid=106610  FOL  neu
│   └── uid=15005  FOL  neu
└── uid=796940  FOL  neu
```

> **Edge tally:** FOL=34  NON=7  UNK=0  →  observed follower ratio: 83%

---

### [17/20] Cascade `17246354` — C6
Size=9 · Depth=2 · Breadth=6 · Follower=0.88 · Sentiment: 11% neg / 89% neu / 0% pos · Event: 0

**Root text:**
> Mein Herz ist bei allen Jüdinnen und Juden, die nach all dem Terror der Hamas in Israel
> nun auch noch die Bilder aus Dagestan sehen müssen, die dieser Tage Angst haben müssen
> auf die Straße zu gehen. Das sollte so nicht sein.

**Cascade tree:**
```
ROOT  uid=106667  sent=neu
├── uid=38988  FOL  neu
│   ├── uid=396819  NON  neu
│   ├── uid=515439  FOL  neu
│   ├── uid=152342  FOL  neu
│   ├── uid=326174  FOL  neu
│   ├── uid=17749  FOL  neu
│   └── uid=518524  FOL  neu
└── uid=332270  FOL  neu
```

> **Edge tally:** FOL=7  NON=1  UNK=0  →  observed follower ratio: 88%

---

### [18/20] Cascade `5138581` — C6
Size=6 · Depth=3 · Breadth=3 · Follower=1.00 · Sentiment: 17% neg / 83% neu / 0% pos · Event: 1

**Root text:**
> The White House plans to combine aid to Ukraine and Israel into one package in the hopes
> that such a pairing would increase the chance that Congress would approve aid to Ukraine
> despite growing opposition from House Republicans.
> www.washingtonpost.com/politics/202....

**Cascade tree:**
```
ROOT  uid=3894  sent=neu
├── uid=295792  FOL  neu
│   └── uid=14853  FOL  neu
│       └── uid=70345  FOL  neu
├── uid=62656  FOL  neu
└── uid=146798  FOL  neu
```

> **Edge tally:** FOL=5  NON=0  UNK=0  →  observed follower ratio: 100%

---

### [19/20] Cascade `12137074` — C6
Size=8 · Depth=6 · Breadth=2 · Follower=0.57 · Sentiment: 12% neg / 88% neu / 0% pos · Event: 0

**Root text:**
> Israel har rett til å forsvare seg. Mot premature babyer.

**Cascade tree:**
```
ROOT  uid=60052  sent=neu
└── uid=61431  NON  neu
    └── uid=60184  NON  neu
        └── uid=61388  NON  neu
            └── uid=60214  FOL  neu
                └── uid=527894  FOL  neu
                    ├── uid=83459  FOL  neu
                    └── uid=730706  FOL  neu
```

> **Edge tally:** FOL=4  NON=3  UNK=0  →  observed follower ratio: 57%

---

### [20/20] Cascade `12062750` — C6
Size=27 · Depth=6 · Breadth=9 · Follower=0.77 · Sentiment: 4% neg / 96% neu / 0% pos · Event: 1

**Root text:**
> Nichts erklärt die Notwendigkeit der Existenz des Staates Israel vorbildlicher als die
> hasserfüllten Kommentare der internationalen Hamas-Unterstützeria

**Cascade tree:**
```
ROOT  uid=84816  sent=neu
├── uid=13880  FOL  neu
│   ├── uid=101582  FOL  neu
│   └── uid=337585  NON  neu
│       ├── uid=471384  FOL  neu
│       └── uid=720870  NON  neu
│           ├── uid=15333  FOL  neu
│           │   ├── uid=153685  FOL  neu
│           │   └── uid=476762  FOL  neu
│           └── uid=142455  NON  neu
│               └── uid=97720  NON  neu
│                   └── uid=1877148  FOL  neu
├── uid=229756  FOL  neu
│   └── uid=1634254  NON  neu
├── uid=8236  FOL  neu
├── uid=145859  FOL  neu
├── uid=876780  FOL  neu
├── uid=44729  FOL  neu
│   └── uid=14602  FOL  neu
├── uid=707045  FOL  neu
│   └── uid=98882  FOL  neu
│       └── uid=173547  FOL  neu
│           └── uid=14882  FOL  neu
├── uid=13229  FOL  neu
└── uid=17836  FOL  neu
    ├── uid=13257  FOL  neu
    └── uid=445598  NON  neu
```

> **Edge tally:** FOL=20  NON=6  UNK=0  →  observed follower ratio: 77%

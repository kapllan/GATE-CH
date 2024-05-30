# special tokens used by llama 2 chat
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

PROMPT_1 = B_INST + ''' Ich gebe dir einen Auszug aus einer Ausschreibung. 
                        Extrahiere folgende Informationen, sofern diese vorhanden sind: Zuschlagskriterien (kriterium), Nummern der Zuschlagskriterien (zkNummer), Gewichtung der Zuschlagskriterien (gewichtung), sowie maximale Punkte der Zuschlagskriterien (maxPunkte). 
                        Strukturiere deine Antwort  in Form einer Json, die wie folgt aufgebaut sein soll: 
                        [
                            {
                            "zkNummer": "",
                            "kriterium": "",
                            "gewichtung": "",
                            "maxPunkte": ""
                            },
                            {
                            "zkNummer": "",
                            "kriterium": "",
                            "gewichtung": "",
                            "maxPunkte": ""
                            }
                        ]

                        Die Json oben ist nur ein Beispiel. 
                        Nicht alle Felder in der Json müssen im Text repräsentiert sein. 
                        Wenn du für einige Felder keine Informationen findest, fügst du einfach einen leeren String ein. 
                        Du musst deine eigene Json auf Grundlage der Ausschreibung, die ich dir gleich zeige, konstruieren.
                        
                        Wenn keine Zuschlagskriterien genannt werden, gibt einfach eine leere Json aus, d.h. eine Json, die so aussieht: [{}]. 
                        WICHTIG: Gebe als Antwort nur eine Json aus und sage sonst nichts weiter! 
                        
                        Hier ist der Auszug:\n\n'''

PROMPT_2 = B_INST + ''' Vorab folgende Hintergrundinformation: Zuschlagskriterien sind etwas anderes als Eignungskriterien.
                        Zuschlagskriterien werden oft mit ZK abgekürzt, Eignungskriterien werden oft mit EZ abgekürzt.
                        Ich gebe dir einen Auszug aus einer Ausschreibung. 
                        Extrahiere folgende Informationen, sofern diese vorhanden sind: Zuschlagskriterien (kriterium), Nummern der Zuschlagskriterien (zkNummer), Gewichtung der Zuschlagskriterien (gewichtung), sowie maximale Punkte der Zuschlagskriterien (maxPunkte). 
                        Strukturiere deine Antwort  in Form einer Json, die wie folgt aufgebaut sein soll: 
                        [
                            {
                            "zkNummer": "",
                            "kriterium": "",
                            "gewichtung": "",
                            "maxPunkte": ""
                            },
                            {
                            "zkNummer": "",
                            "kriterium": "",
                            "gewichtung": "",
                            "maxPunkte": ""
                            }
                        ]

                        Die Json oben ist nur ein Beispiel. 
                        Nicht alle Felder in der Json müssen im Text repräsentiert sein. 
                        Wenn du für einige Felder keine Informationen findest, fügst du einfach einen leeren String ein. 
                        Du musst deine eigene Json auf Grundlage der Ausschreibung, die ich dir gleich zeige, konstruieren.
                        
                        Wenn keine Zuschlagskriterien genannt werden, gibt einfach eine leere Json aus, d.h. eine Json, die so aussieht: [{}]. 
                        WICHTIG: Gebe als Antwort nur eine Json aus und sage sonst nichts weiter! 
                        
                        Hier ist der Auszug:\n\n'''

PROMPT_3 = B_INST + ''' Vorab folgende Hintergrundinformation: Zuschlagskriterien sind etwas anderes als Eignungskriterien.
                        Zuschlagskriterien werden oft mit ZK abgekürzt, Eignungskriterien werden oft mit EZ abgekürzt.
                        Ich gebe dir einen Auszug aus einer Ausschreibung. 
                        Extrahiere NUR die Zuschlagskriterien (kriterium). 
                        Strukturiere deine Antwort  in Form einer Json, die wie folgt aufgebaut sein soll: 
                        [
                            {
                            "zkNummer": "",
                            "kriterium": "",
                            "gewichtung": "",
                            "maxPunkte": ""
                            },
                            {
                            "zkNummer": "",
                            "kriterium": "",
                            "gewichtung": "",
                            "maxPunkte": ""
                            }
                        ]

                        Du musst nur das Feld 'kriterium' befüllen, die anderen Felder in der Json bleiben leer.
                        Wenn keine Zuschlagskriterien genannt werden, gibt einfach eine leere Json aus, d.h. eine Json, die so aussieht: [{}].
                        Hier einige Infos, wie man Zuschlagskriteriterien gut erkennt: Sie werden oft mit ZK abgekürzt und haben oft eine Gewichtung in Prozent. Außerdem muss das Wort Zuschlagskriterium im Text vorkommen, da wir nur explizite Angaben extrahieren.
                        WICHTIG: Gebe als Antwort nur eine Json aus und sage sonst nichts weiter! 

                        Hier ist der Auszug:\n\n'''

PROMPT_4 = B_INST + ''' Hier einige grundlegende Informationen zu Ausschreibungen.
                        Zuschlagskriterien werden oft mit ZK abgekürzt und haben oft, aber nicht immer, eine Gewichtung, maximale Punktzahl und eine Nummer.
                        Die Gewichtung wird immer in Prozent (%) angegeben.
                        Die maximale Punktzahl ist eine Nummer.
                        Die Nummer des Zuschlagskriteriums fängt oft, aber nicht immer, mit der Abkürtung ZK an.       
                        Ich gebe dir einen Auszug aus einer Ausschreibung. 
                        Wenn es in dem Auszug um Zuschlagskriterien geht, extrahiere folgende Informationen, sofern diese explizit im Auszug genannt werden: Zuschlagskriterien (kriterium), Nummern der Zuschlagskriterien (zkNummer), Gewichtung der Zuschlagskriterien (gewichtung), sowie maximale Punkte der Zuschlagskriterien (maxPunkte). 
                        Strukturiere deine Antwort  in Form einer Json, die wie folgt aufgebaut sein soll: 
                        [
                            {
                            "zkNummer": "",
                            "kriterium": "",
                            "gewichtung": "",
                            "maxPunkte": ""
                            },
                            {
                            "zkNummer": "",
                            "kriterium": "",
                            "gewichtung": "",
                            "maxPunkte": ""
                            }
                        ]

                        Die Json oben ist nur ein Beispiel. 
                        Nicht alle Felder in der Json müssen im Text repräsentiert sein. 
                        Wenn du für einige Felder keine Informationen findest, fügst du einfach einen leeren String ein. 
                        Du musst deine eigene Json auf Grundlage der Ausschreibung, die ich dir gleich zeige, konstruieren.

                        Wenn keine Zuschlagskriterien genannt werden, gibt einfach eine leere Json aus, d.h. eine Json, die so aussieht: [{}]. 
                        WICHTIG: Gebe als Antwort nur eine Json aus und sage sonst nichts weiter! 

                        Hier ist der Auszug:\n\n'''

PROMPT_5 = B_INST + ''' Hier einige grundlegende Informationen zu Ausschreibungen.
                        Zuschlagskriterien werden oft mit ZK abgekürzt und haben oft, aber nicht immer, eine Gewichtung, maximale Punktzahl und eine Nummer.
                        Die Gewichtung wird immer in Prozent (%) angegeben.
                        Die maximale Punktzahl ist eine Nummer.
                        Die Nummer des Zuschlagskriteriums fängt oft, aber nicht immer, mit der Abkürtung ZK an.       
                        Ich gebe dir einen Auszug aus einer Ausschreibung. 
                        
                        Erste Aufgabe: Schau erst, ob im Auszug überhaupt Zuschlagskriterien genannt werden. 
                        Wenn keine Zuschlagskriterien genannt werden, gibt einfach eine leere Liste aus, d.h. eine Json, die so aussieht: [{}]. 
                        Wenn aber Zuschlagskriterien genannt werden, und nur dann, machst du weiter mit Aufgabe 2. 
                        
                        Aufgabe 2: Wenn es in dem Auszug um Zuschlagskriterien geht, extrahiere folgende Informationen, sofern diese explizit im Auszug genannt werden: Zuschlagskriterien (kriterium), Nummern der Zuschlagskriterien (zkNummer), Gewichtung der Zuschlagskriterien (gewichtung), sowie maximale Punkte der Zuschlagskriterien (maxPunkte). 
                        Strukturiere deine Antwort  in Form einer Json, die wie folgt aufgebaut sein soll: 
                        [
                            {
                            "zkNummer": "",
                            "kriterium": "",
                            "gewichtung": "",
                            "maxPunkte": ""
                            },
                            {
                            "zkNummer": "",
                            "kriterium": "",
                            "gewichtung": "",
                            "maxPunkte": ""
                            }
                        ]

                        Die Json oben ist nur ein Beispiel. 
                        Nicht alle Felder in der Json müssen im Text repräsentiert sein. 
                        Wenn du für einige Felder keine Informationen findest, fügst du einfach einen leeren String ein. 
                        Du musst deine eigene Json auf Grundlage der Ausschreibung, die ich dir gleich zeige, konstruieren.

                        
                        WICHTIG: Gebe als Antwort nur eine Json aus und sage sonst nichts weiter! 

                        Hier ist der Auszug:\n\n'''

prompt_check_if_award_criteria_1 = B_INST + """Ich gebe dir einen Auszug aus einer Ausschreibung.
                                        Wenn in dem Auszug Zuschlagskriterien genannt werden, sag nur 'Ja'.
                                        Wenn in dem Auszug keine Zuschlagskriterien genannt werden, sag nur 'Nein'.
                                        Sag sonst nichts weiter. 
                                        
                                        Hier ist der Auszug:\n\n
                                        """




prompt_dict = dict()
prompt_dict['prompt_1'] = PROMPT_1
prompt_dict['prompt_2'] = PROMPT_2
prompt_dict['prompt_3'] = PROMPT_3
prompt_dict['prompt_4'] = PROMPT_4
# prompt_dict['prompt_5'] = PROMPT_5


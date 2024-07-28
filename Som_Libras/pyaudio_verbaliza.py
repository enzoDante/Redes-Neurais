import speech_recognition as sr
import vlibras_translate

tradutor = vlibras_translate.translation.Translation()

mic = sr.Recognizer()
with sr.Microphone() as source:
    mic.adjust_for_ambient_noise(source)
    print("Teste de voz")
    audio = mic.listen(source)
    try:
        # frase = mic.recognize_google_cloud(audio_data=audio, language="pt-BR")
        frase = mic.recognize_google(audio,language="pt-BR")

        print(f"Você disse: {frase}")
        
        ptbr_teste = tradutor.preprocess_pt(frase)
        glosa = tradutor.rule_translation(ptbr_teste) 
        print(glosa)
    except sr.UnknownValueError:
        print(f"erro n entendi\n ")



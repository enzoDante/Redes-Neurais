import vlibras_translate
tradutor = vlibras_translate.translation.Translation()
#três vira 3
ptbr_teste = tradutor.preprocess_pt('Maria comprou por três parcelas de 35,50 reais naquela loja hje.')
glosa = tradutor.rule_translation(ptbr_teste) 
print(glosa)
#biblioteca para separar as palavras e serem utilizadas em boneco de libras
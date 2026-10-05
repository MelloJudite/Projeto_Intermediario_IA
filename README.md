# Ativida de robótica para a disciplina de IA de C. Comp

Inteligência Artificial e Robótica — Insper — 2026.2

1. DESCRIÇÃO

Projeto de planejamento de caminhos para um robô em um ambiente parcialmente
conhecido. A proposta é calcular trajetórias a partir de um mapa no formato PGM
e das coordenadas de início e destino, respeitando os obstáculos e mantendo
uma distância segura das paredes.

2. INTEGRANTES

- Judite Rangel de Mello
- Maria Eduarda dos Santos
- Mariana Caetano Machado

3. OBJETIVO E FUNCIONAMENTO ESPERADO

O algoritmo deve receber:
- Um mapa no formato PGM.
- As coordenadas da posição inicial do robô.
- As coordenadas do destino desejado.

A saída deve ser um caminho que permita avançar em direção ao destino dentro
da região conhecida e segura do mapa.

Quando o destino estiver além da região conhecida, o robô deverá seguir até
o limite seguro dessa região e parar. Após perceber novas partes do ambiente,
deverá atualizar o planejamento e calcular um novo caminho. Esse processo
se repete até alcançar o destino, caso exista uma rota viável.

4. REQUISITOS DO PROJETO

- Considerar paredes e demais obstáculos no planejamento.
- Evitar colisões e trajetórias que façam o robô raspar nas paredes.
- Manter uma margem de segurança compatível com as dimensões do robô.
- Buscar uma movimentação suave, evitando movimentos abruptos desnecessários.
- Validar o caminho em simulação antes de realizar testes no robô físico.
- Considerar passagens estreitas e obstáculos adicionais nos testes.

5. BASE DO PROJETO

Arquivo indicado para implementação: astar.py.
O template também disponibiliza imagens para testes simulados.

Detalhes da implementação da equipe:
- Algoritmo e heurística utilizados: A* com fila de prioridade (heapq) e heurística de distância euclidiana. A busca considera oito direções, com custo de 1 para movimentos horizontais e verticais e √2 para diagonais. O custo também inclui uma penalização pela proximidade de paredes e um acréscimo de 0,1 para células desconhecidas.

- Tratamento de obstáculos e margem de segurança: As paredes são bloqueadas, e movimentos diagonais não podem atravessar cantos entre obstáculos. Um campo de potencial, calculado a partir da distância até as paredes, favorece trajetórias mais afastadas delas. Na simplificação e suavização, os segmentos propostos são verificados com uma distância mínima de 4 pixels das paredes. Essa margem não é uma restrição obrigatória na busca A*, portanto o código não garante que todo o caminho mantenha esse afastamento.

- Estratégia de replanejamento: A busca pode considerar regiões desconhecidas, mas a função know_path limita o caminho ao trecho inicial conhecido, interrompendo-o antes da primeira célula não livre. O replanejamento depende de uma nova execução com o mapa atualizado e a posição atual do robô. O código apresentado não implementa automaticamente a atualização do mapa, a parada física ou o ciclo de replanejamento.

- Tratamento dos pontos do caminho e suavidade do movimento: Primeiro, são removidos os pontos intermediários que não representam mudanças de direção. Depois, são buscadas ligações diretas entre pontos mais distantes, verificadas pela função _segmento_seguro. Por fim, são aplicadas três iterações de uma adaptação do método de Chaikin para arredondar as mudanças de direção. O resultado pode conter segmentos diagonais e coordenadas decimais. A suavização busca reduzir movimentos abruptos, mas exige validação final da trajetória, pois nem todos os segmentos resultantes são novamente verificados.


6. VÍDEO DE DEMONSTRAÇÃO

Link do vídeo do robô funcionando no labirinto:https://youtube.com/shorts/2__AHyJJvbI?feature=share


7. CRITÉRIOS DE AVALIAÇÃO

A: Funciona no robô físico, mantém distância das paredes, apresenta movimento
   suave e consegue atravessar passagens estreitas com obstáculos adicionais.
B: Funciona no robô físico, mantém distância das paredes e apresenta movimento
   suave, mas não consegue atravessar os trechos mais estreitos com obstáculos.
C: Funciona no robô físico, porém passa muito próximo das paredes e apresenta
   diversos movimentos abruptos.
D: Funciona em simulação, com um caminho visível e aparentemente coerente.
I: A equipe trabalhou no projeto, mas o código não funciona em simulação.

Cada integrante deve estar presente em pelo menos dois dos três dias de
atividade; caso contrário, receberá conceito I, independentemente do resultado.
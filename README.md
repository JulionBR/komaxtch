# komaxtch

Converte mangá e HQ (EPUB, CBZ ou pasta de imagens) em **XTCH** para ler **quadro a quadro**
no Xteink X3/X4 com o firmware [CrossPoint](https://github.com/crosspoint-reader/crosspoint-reader).

Numa tela de 3,7" a página inteira fica ilegível. O komaxtch detecta os quadros, coloca cada um
na ordem de leitura (direita→esquerda no mangá) e amplia para ocupar a tela. Opcionalmente, também cria telas extras com zoom nos balões de fala pequenos.

## Instalação

```bash
brew install uv
uv tool install .
uv tool update-shell   # coloca ~/.local/bin no PATH (uma vez)
```

## Uso

```bash
komaxtch convert "One Piece 01.epub"   # gera "One Piece 01.xtch" ao lado
komaxtch convert ~/Mangas/             # converte todos os EPUB/CBZ da pasta
```

Copie o `.xtch` para o cartão SD do leitor e abra pelo CrossPoint.

### Modos

| `--mode` | O que cada tela mostra |
|---|---|
| `auto` (padrão) | um quadro por tela; quando o quadro aparece muito reduzido (página inteira, splash), vem em seguida o zoom nos balões |
| `panel` | só um quadro por tela |
| `bubble` | o quadro e o zoom em todo balão que ficaria pequeno (mais telas) |
| `page` | a página inteira |

Para comparar os modos antes de converter um volume inteiro:

```bash
komaxtch preview livro.epub --pages 1-5 -o preview/
```

Cada página gera um `comparacao_pNNN.png` com as telas de cada modo lado a lado, simulando os
4 tons de cinza do aparelho.

### Opções úteis

- `--direction auto|rtl|ltr`: o padrão `auto` usa o metadado do livro. Sem metadado, usa `ltr`.
- `--device x3|x4`: X3 = 528×792 (padrão), X4 = 480×800.
- `--rotate none|cw|ccw`: por padrão nada é girado, para você não ter que virar o aparelho. `cw`/`ccw` giram os quadros largos para ficarem maiores.
- `--dither floyd|none`: `none` deixa o traço mais seco.
- `--no-split-spreads`: não divide páginas duplas.

## Detecção

Cada página passa por uma cadeia de detectores, do mais rápido ao mais caro:

1. **OpenCV (local, instantâneo):** acha os quadros pelas calhas brancas, corta quadros
   grudados (balão atravessando a calha, arte sangrada separada por faixas brancas) e dá uma
   nota de confiança. Num volume típico, resolve cerca de 60% das páginas sozinho.
2. **Gemini (camada gratuita), recomendado:** recebe só as páginas com nota baixa (blocos
   enormes, splash, quadros grudados que não deu para cortar). As caixas que ela devolve
   ainda passam pelo corte de calhas do OpenCV, para separar colunas dentro de faixas.
   Crie uma chave grátis no [Google AI Studio](https://aistudio.google.com/apikey) e coloque-a
   em `~/.config/komaxtch/.env` (vale de qualquer pasta) ou num `.env` na pasta atual:

   ```bash
   GEMINI_API_KEY=sua-chave
   # GEMINI_MODELS=gemini-3.5-flash,gemini-3.1-flash-lite
   ```

   **Cota gratuita.** Em out/2026 a camada grátis permitia 20 requisições por dia, por
   modelo. Para caber nisso, o komaxtch:
   - manda 8 páginas por requisição (um volume de mangá gasta umas 10 requisições);
   - desliga o raciocínio do modelo (metade dos tokens, resposta em ~3 s por página);
   - espera o tempo que a API pede quando estoura o limite por minuto;
   - quando a cota diária de um modelo acaba, passa para o próximo da lista e anota em
     `~/.cache/komaxtch/quota.json` até quando ele está bloqueado;
   - se todos acabarem, termina a conversão com o palpite local e avisa a partir de quando
     rodar de novo. As páginas já analisadas ficam em cache: a nova execução só pede as que
     faltaram.

   Na camada gratuita, o Google pode usar as imagens enviadas para melhorar os produtos
   dele. Para desligar, use `--no-cloud`.
3. **IA local via Ollama, opcional (`--local-ai`):** grátis e offline, mas o modelo Qwen3-VL 4B
   ocupa ~4 GB de RAM e leva de 20 a 25 s por página. Num Mac de 8 GB isso deixa o sistema
   quase travado durante a conversão, por isso fica desligado por padrão.

   ```bash
   brew install ollama
   brew services start ollama
   ollama pull qwen3-vl:4b-instruct
   ```

4. Sem resposta confiável, a página sai inteira (ou com o melhor palpite do OpenCV).

A ordem de leitura é sempre calculada pelo komaxtch (cortes nas calhas, direita→esquerda no
mangá), porque a ordem devolvida pelos modelos não é confiável. As respostas das IAs ficam em
cache (`~/.cache/komaxtch`), então reconverter é rápido e não gasta cota.

Os balões (modo `bubble`) são detectados localmente pelo YOLOv8
[`ogkalu/comic-speech-bubble-detector-yolov8m`](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m)
(Apache-2.0). Os pesos são baixados na primeira execução.

As respostas da IA são limpas antes de usar: tiras finas e caixas repetidas são descartadas e,
se a resposta vier com caixas se cobrindo, a página fica com o palpite local (ou inteira, com
zoom nos balões no modo `auto`). Um corte só acontece se os dois lados tiverem tamanho de quadro,
para não picotar onomatopeias nem cortar balões.

Para ver o que foi detectado:

```bash
komaxtch debug livro.epub --pages 1-5 -o debug/
```

## Formato XTCH

O gravador segue o parser do firmware (`lib/Xtc/Xtc/XtcTypes.h` e `XtcReaderActivity.cpp`):

- cabeçalho de 56 bytes
- título em `0x38` e autor em `0xB8`
- índice de 16 bytes por página
- páginas XTH com 2 bit-planes em colunas da direita para a esquerda

Para validar um arquivo:

```bash
komaxtch inspect livro.xtch --extract telas/
```

## Desenvolvimento

```bash
uv sync
uv run pytest
PYTHONPATH=src uv run --no-sync python -m komaxtch --help
```

No macOS, o `uv` marca a `.venv` como oculta, e o Python 3.12 ignora arquivos `.pth` ocultos.
Isso quebra a instalação editável. Por isso, em desenvolvimento, o CLI roda com
`PYTHONPATH=src`. O pytest já acha o código em `src/` sozinho. Depois de mudar o código,
atualize o comando instalado com `uv tool install --force .`.

## Licença

Copyright (C) 2026 Julio César Da Silva

O komaxtch é software livre: você pode usar, estudar, modificar e redistribuir sob os termos da
[GNU Affero General Public License v3.0](LICENSE) ou (a seu critério) qualquer versão posterior.
Quem distribuir uma versão modificada, ou oferecê-la como serviço pela rede, precisa publicar o
código dela sob a mesma licença.

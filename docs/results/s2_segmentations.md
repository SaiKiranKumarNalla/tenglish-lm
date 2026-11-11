| spelling | BPE | Unigram | Sarvam-2B (pretrained) |
|---|---|---|---|
| `enti` | `▁enti` | `▁enti` | `▁ent` `i` |
| `yenti` | `▁yenti` | `▁yenti` | `▁y` `enti` |
| `anti` | `▁anti` | `▁anti` | `▁anti` |
| `entra` | `▁entra` | `▁entra` | `▁ent` `ra` |
| `yentra` | `▁yentra` | `▁yentra` | `▁y` `ent` `ra` |
| `chesanu` | `▁chesanu` | `▁chesanu` | `▁c` `hes` `an` `u` |
| `chesaanu` | `▁chesaanu` | `▁chesaanu` | `▁c` `hes` `a` `an` `u` |
| `chesa` | `▁chesa` | `▁chesa` | `▁c` `hes` `a` |
| `intha` | `▁intha` | `▁intha` | `▁in` `th` `a` |
| `inta` | `▁inta` | `▁inta` | `▁int` `a` |

Unigram can **sample** different segmentations of the same word (subword regularization):

- `▁velli po t h unna n u`
- `▁vellipo t h un na n u`
- `▁vel li p o th u n na n u`
- `▁ve l l ipo t h u nna n u`

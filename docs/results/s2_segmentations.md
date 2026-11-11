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

- `▁velli p o th u n na nu`
- `▁velli p o thu nna nu`
- `▁vellipo thunna n u`
- `▁vellipo th u nna nu`

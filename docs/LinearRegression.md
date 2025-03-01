# Regresja liniowa

Zanim przejdziemy do właściwej zabawy sieciami neuronowymi sprawdźmy jak działa jedna z podstawowych koncepcji z nimi związana: uczenie się za pomocą wstecznej propagacji błędu[^wiki_propagacja_wsteczna].

Zbudujemy sobie zatem coś, co jeszcze nie jest siecią neuronową, ale - tak jak ona - potrafi się uczyć.

Załóżmy, że jesteśmy w posiadaniu tajemniczych danych:

| x<sub>1</sub> | x<sub>2</sub> |    y |
|		   ---: |          ---: | ---: |
|             1	|             2 |   -3 |
|			  2 |             3 |   -5 |
|             3 |             4 |   -7 |
|             4 |             5 |   -9 |
|             5 |             6 |  -11 |
|             0 |             0 |    0 |
|            -5 |            -6 |   11 |

Aby przypadkiem gdzieś nie zniknęły zapiszmy sobie je w ten sposób:

```csharp
Matrix xTrain = new(new float[,] { { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 5 }, { 5, 6 }, { 0, 0 }, { -5, -6 } });
Matrix yTrain = new(new float[,] { { -3 }, { -5 }, { -7 }, { -9 }, { -11 }, { 0 }, { 11 } });
```

Naszym celem będzie znalezienie takiej funkcji liniowej, która najlepiej odwzoruje zależność między x<sub>1</sub> i x<sub>2</sub> a y. Funkcja ta będzie miała postać: 

y = a * x~1~ + b * x~2~ + c,

gdzie wartości a, b i c będą przedmiotem naszych niestrudzonych poszukiwań.

Zbudujemy w tym celu coś, co nazywa się **modelem regresji liniowej**[^wiki_regresja_liniowa] i *wytrenujemy* go używając wyżej podanych danych. Pod pojęciem *wytrenowanie* rozumiem znalezienie takich wartości parametrów a, b i c, które minimalizują błąd predykcji modelu. Błąd ten będziemy mierzyć za pomocą funkcji kosztu w postaci sumy kwadratów różnic między wartościami przewidywanymi przez model a wartościami rzeczywistymi.

Czyli po ludzku: jeżeli model uzna, że dla x~1~ = 1 i x~2~ = 2 wartość y wynosi -5 (bo tak akurat sobie wymyślił), a w rzeczywistości wartość ta wynosi -3 (patrz powyższa tabelka), to model popełnił błąd równy -5 - (-3) = -2, a funkcja kosztu zwiększy w tym przypadku swoją wartość o (-2)^2^, czyli o 4. Jeżeli dla kolejnego zestawu danych (x~1~ = 2, x~2~ = 3) model uzna, że y = -4, a w rzeczywistości y wynosi -5, to model popełni błąd równy -4 - (-5) = 1, a funkcja kosztu zwiększy się o kolejne 1^2^ = 1.

Jeżeli teraz zsumujemy wszystkie koszty dla wszystkich naszych danych, to otrzymamy wartość funkcji kosztu, którą będziemy chcieli zminimalizować, czyli będziemy dążyli do tego, żeby jej wartość wyniosła 0.

$\sqrt{3x-1}+(1+x)^2$

## Przypisy

[^wiki_propagacja_wsteczna]: https://pl.wikipedia.org/wiki/Propagacja_wsteczna
[^wiki_regresja_liniowa]: https://pl.wikipedia.org/wiki/Regresja_liniowa

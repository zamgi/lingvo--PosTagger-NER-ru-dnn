var LOCALSTORAGE_KEY           = 'Lingvo.PosTagger.',
    LOCALSTORAGE_TEXT_KEY      = LOCALSTORAGE_KEY + getRegimenModelType() + '.text',
    LOCALSTORAGE_MODELTYPE_KEY = LOCALSTORAGE_KEY + getRegimenModelType() + '.modelType',

    DEFAULT_TEXT_POS =
'Эти типы стали есть в цехе.\r\n' +
'\r\n' +
'Вася, маша руками и коля дрова, морочил голову.\r\n' +
'Вася, маша и коля пошли гулять.\r\n' +
'\r\n' +
'Маша руками мыла посуду.\r\n' +
'Сильно маша руками вася звал на помощь.\r\n' +
'Реки стали красными.\r\n' +
'Реки стали красными потоками текли.\r\n' +
'\r\n' +
'Гло́кая ку́здра ште́ко будлану́ла бо́кра и курдя́чит бокрёнка.\r\n' +
'Варкалось. Хливкие шорьки пырялись по наве, и хрюкотали зелюки, как мюмзики в мове.\r\n' +
'\r\n' +
'В Петербурге перед судом предстанет высокопоставленный офицер Генерального штаба ВС РФ. СКР завершил расследование уголовного дела против главы военно-топографического управления Генштаба контр-адмирала Сергея Козлова, обвиняемого в превышении должностных полномочий и мошенничестве.\r\n' +
'"Следствием собрана достаточная доказательственная база, подтверждающая виновность контр-адмирала Козлова в инкриминируемых преступлениях, в связи с чем уголовное дело с утвержденным обвинительным заключением направлено в суд для рассмотрения по существу", - рассказали следователи.\r\n' +
'Кроме того, по инициативе следствия представителем Минобороны России к С.Козлову заявлен гражданский иск о возмещении причиненного государству ущерба на сумму свыше 27 млн руб.\r\n' +
'По данным следователей, в июле 2010г. военный чиновник отдал подчиненному "заведомо преступный приказ" о заключении лицензионных договоров с компаниями "Чарт-Пилот" и "Транзас". Им необоснованно были переданы права на использование в коммерческих целях навигационных морских карт, являвшихся интеллектуальной собственностью РФ. В результате ущерб составил более 9,5 млн руб.\r\n' +
'Контр-адмирал также умолчал о наличии у него в собственности квартиры в городе Истра Московской области. В результате в 2006г. центральной жилищной комиссии Минобороны и Управления делами президента РФ С.Козлов был признан нуждающимся в жилье и в 2008г. получил от государства квартиру в Москве площадью 72 кв. м и стоимостью 18,5 млн руб. Квартиру позднее приватизировала его падчерица.\r\n' +
        'Против С. Козлова возбуждено дело по п."в" ч.3 ст.286 (превышение должностных полномочий, совершенное с причинением тяжких последствий) и ч.4 ст.159 (мошенничество, совершенное в особо крупном размере) УК РФ.\r\n',

    DEFAULT_TEXT_NER = 'венгрия готова оплачивать российский газ рублями, заявил глава местного мид петер сийярто. ранее с требованием платить за газ из россии в рублях выступал президент рф владимир путин.\r\n' +
'\r\n' +
'«что касается оплаты в рублях, то у нас есть решение, которое позволит не нарушать какие-либо санкции, но в то же время обеспечит поставки газа в венгрию», — заявил сийярто. его выступление опубликовано на странице министра в соцсетях.\r\n' +
'\r\n' +
'о том, что венгрия пошла на требование россии писало агентство reuters со ссылкой на слова премьер-министра страны виктора орбана. тогда он заявлял, что «это не является проблемой».\r\n' +
'\r\n' +
'россия проводит на украине специальную операцию по демилитаризации и денацификации. президент россии владимир путин заявил, что это вынужденное решение, необходимое для защиты жителей донбасса от многолетнего геноцида. после начала спецоперации евросоюз ввел против россии масштабные экономические санкции. путин заявил, что россия в ответ будет продавать газ «недружественным» странам за рубли. ряд стран, например, германия и франция отказались выполнять ультиматум.\r\n' +
'\r\n' +
'продолжайте получать новости ura.ru даже в случае блокировки google, подпишитесь на telegram-канал ura.ru.\r\n' +
'\r\n' + 
'канцлер австрии карл нехаммер приедет в москву, чтобы обсудить с президентом россии владимиром путиным ситуацию на украине. об этом сообщил пресс-секретарь президента рф дмитрий песков в ходе конференц-колла.\r\n' +
'\r\n' +
'«разговор будет идти о ситуации вокруг украины. все остальные вопросы лучше адресовать австрийской стороне. закрытый формат — ее инициатива», — передает слова пескова корреспондент ura.ru.\r\n' +
'\r\n' +
'глава мид австрии александер шалленберг ранее заявил, что лидеры проведут переговоры «один на один», пишет rt. журналистов не пустят на встречу, пресс-конференции по итогам переговоров также не ожидается.\r\n' +
'\r\n' +
'россия в конце февраля начала специальную операцию по демилитаризации и денацификации украины. владимир путин заявил, что целью спецоперацию является освобождение жителей донбасса от многолетнего геноцида. согласно данным опроса вциом, 79% респондентов поддержали решение президента. в минобороны рф подчеркнули, что приоритетом российской армии является недопущение жертв среди мирного населения.\r\n' +
'\r\n' +
'подписывайтесь на ura.ru в яндекс.новости и на наш канал в яндекс.дзен, следите за главными новостями россии и урала в telegram-канале ura.ru и получайте все самые важные известия с доставкой в вашу почту в нашей ежедневной рассылке.\r\n',

    DEFAULT_TEXT = ((getRegimenModelType() || '').toLowerCase() === 'ner') ? DEFAULT_TEXT_NER : DEFAULT_TEXT_POS
    ;
$(document).ready(function () {
    var _Last_Responce;
    var textOnChange = function () { $('#textLength').html('text length: ' + $('#text').val().length.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ') + ' chars'); };
    var getText = function ($text) { var text = trim_text($text.val().toString()); if (is_text_empty(text)) { alert('Enter text for PosTagger.'); $text.focus(); return (null); } return (text); };

    $('#text').focus(textOnChange).change(textOnChange).keydown(textOnChange).keyup(textOnChange).select(textOnChange).focus();
    $(window).resize(function () { $('#processResult').height($(window).height() - $('#processResult').position().top - 5/*25*/); }).trigger('resize');
    $('#posStyleCheckbox').prop('checked', !!localStorage.getItem(LOCALSTORAGE_KEY + 'posStyleCheckbox')).change(function () {
        var $this = $(this), ch = $this.is(':checked'), k = LOCALSTORAGE_KEY + 'posStyleCheckbox',
            pos_v1 = '/pos_tagger_v1.css'.toLowerCase(), pos_v2 = '/pos_tagger_v2.css'.toLowerCase(),
            ner_v1 = '/ner_v1.css'       .toLowerCase(), ner_v2 = '/ner_v2.css'       .toLowerCase();
        if (ch) localStorage.setItem(k, true); else localStorage.removeItem(k);

        var disabled_pos = ch ? pos_v2 : pos_v1,
            enabled_pos  = ch ? pos_v1 : pos_v2,
            disabled_ner = ch ? ner_v2 : ner_v1,
            enabled_ner  = ch ? ner_v1 : ner_v2;
        var $disabled_pos_ss = $(document.styleSheets).filter(function (_, ss) { return (ss.href.toLowerCase().indexOf(disabled_pos) !== -1); }),
            $enabled_pos_ss  = $(document.styleSheets).filter(function (_, ss) { return (ss.href.toLowerCase().indexOf(enabled_pos ) !== -1); }),
            $disabled_ner_ss = $(document.styleSheets).filter(function (_, ss) { return (ss.href.toLowerCase().indexOf(disabled_ner) !== -1); }),
            $enabled_ner_ss  = $(document.styleSheets).filter(function (_, ss) { return (ss.href.toLowerCase().indexOf(enabled_ner ) !== -1); });
        if ($disabled_pos_ss.length) $disabled_pos_ss[0].disabled = true;
        if ($enabled_pos_ss .length) $enabled_pos_ss [0].disabled = false;
        if ($disabled_ner_ss.length) $disabled_ner_ss[0].disabled = true;
        if ($enabled_ner_ss .length) $enabled_ner_ss [0].disabled = false;

        if (_Last_Responce) {
            usePosTaggerStyle(_Last_Responce);
        }
    }).trigger('change').parent('div').parent('div').show();
    function isUsePosTaggerStyle_PlainLayout() { return (!$('#posStyleCheckbox').is(':checked')); };

    (function () {
        $('#text').val(localStorage.getItem(LOCALSTORAGE_TEXT_KEY) || DEFAULT_TEXT).focus();

        $.get('/PosTagger/GetModelInfoKeys?regimenModelType=' + encodeURIComponent(getRegimenModelType()) + '&' + Math.random())
         .done(function (resp) {
             if (!resp.errorMessage && resp.length) {
                 var $mt = $('#modelType').empty();
                 for (var i = 0, len = resp.length; i < len; i++) {
                     var mt = resp[i], t = mt;//.replaceAll('_', ' ').replaceAll('ru', '').trim();
                     $mt.append( $('<option>').attr('value', mt).text( t ) );
                 }

                 $mt.val( localStorage.getItem( LOCALSTORAGE_MODELTYPE_KEY ) );
                 if ( !$mt.val() ) $mt.val( $mt.find('option').val() );
             }
         });
    })();
    $('#resetText2Default').click(function () { $('#text').val(''); setTimeout(function () { $('#text').val(DEFAULT_TEXT).focus(); }, 100); });

    $('#mainPageContent').on('click', '#processButton', function () {
        if ($(this).hasClass('disabled')) return (false);

        var text = getText($('#text'));
        if (!text) return (false);

        processing_start();
        var modelType = $('#modelType').val();
        if (text !== DEFAULT_TEXT) localStorage.setItem(LOCALSTORAGE_TEXT_KEY, text); else localStorage.removeItem(LOCALSTORAGE_TEXT_KEY);
        localStorage.setItem(LOCALSTORAGE_MODELTYPE_KEY, modelType);

        _Last_Responce = null;
        $('#processResult table').empty();
        $.ajax({
            type: 'POST', contentType: 'application/json; charset=utf-8',
            url: '/PosTagger/Run', data: JSON.stringify({ text: text, modelType: modelType }),
            error: function () { processing_end_with_error('Server error.'); },
            success: function (resp) {
                if (resp.error && resp.error.errorMessage) { processing_end_with_error(resp.error.errorMessage + '          (' + resp.error.fullErrorMessage + ')'); return; }
                if (!resp.sents || !resp.sents.length) { processing_end_with_undefined(); return; }

                _Last_Responce = resp;
                processing_end_without_error();
                usePosTaggerStyle(resp);
            }
        });
    });

    function get_title_4_pos(/*label*/) { return ''/*label*/; };
    function get_title_4_ner(label) {
        switch (label) {
            case 'B-LOC': case 'I-LOC': return ('Location/Geo');
            case 'B-ORG': case 'I-ORG': return ('Organization');
            case 'B-PER': case 'I-PER': return ('Person');
            case 'O': return ('Other');
            default: return (label);
        }
    };
    function usePosTaggerStyle(resp) {
        if (isUsePosTaggerStyle_PlainLayout()) {
            usePosTaggerStyle_PlainLayout(resp);
        } else {
            usePosTaggerStyle_TableLayout(resp);
        }
    }
    function usePosTaggerStyle_TableLayout(resp) {
        var $result = $('#processResult table').empty(),
            is_ner = (getRegimenModelType() === 'Ner'),
            header = '<tr>' +
                       '<th>original-word</td>' +
                       '<th>' + (is_ner ? 'named-entity-type' : 'part-of-speech') + '</td>' +
                     '</tr>',
            get_title = (is_ner ? get_title_4_ner : get_title_4_pos),
            get_title_2 = (is_ner ? get_title_4_ner : function (label) { return label; });
            get_label_text = (is_ner ? function (label) { return (label === 'O' ? '-' : label); } : function (label) { return label; });
        var trs = [header];
        for (var i = 0, len = resp.sents.length ; i < len; i++) {
            var o = resp.sents[i],
                sentText   = o.tuples.map(t => t.word).join(' '),
                sentNumber = (i + 1),
                even_odd   = (((sentNumber % 2) === 0) ? 'even' : 'odd');

            var $tr = $('<tr>').addClass(even_odd);
                $('<td>').attr('colspan', '2').css('text-align', 'center').append( $('<span>').addClass('sent-number').text(sentNumber + '). ').append( $('<i>').text( sentText ) ) ).appendTo( $tr );
            trs.push( $tr[ 0 ].outerHTML );
            if (o.error && o.error.errorMessage) {
                trs.push("<tr>" + $('<div>').addClass('error bold').text(o.error.errorMessage + '          (' + o.error.fullErrorMessage + ')')[ 0 ].outerHTML + "</tr>" );
            } else {
                for (var j = 0, ts = o.tuples, len_2 = ts.length; j < len_2; j++) {
                    var t = ts[j];

                    $tr = $('<tr>').addClass(even_odd);
                    $('<td>').append( $('<span>').text(t.word).attr('title', get_title_2( t.label ) + ': \'' + t.word + '\'') ).appendTo($tr);
                    $('<td>').append($('<span>').text(get_label_text( t.label )).addClass(t.label).attr('label', t.label).attr('title', get_title( t.label )) ).appendTo($tr);
                    trs.push( $tr[ 0 ].outerHTML );
                }
            }
            trs.push('<tr><td colspan="2"/></tr>');
        }
        var html = trs.join('');//.replaceAll('\r\n', '<br/>').replaceAll('\n', '<br/>').replaceAll('\t', '&nbsp;&nbsp;&nbsp;&nbsp;');
        $result.html(html);
    };
    function usePosTaggerStyle_PlainLayout(resp) {
        var $result = $('#processResult table').empty(),
            is_ner = (getRegimenModelType() === 'Ner'),
            get_title = (is_ner ? get_title_4_ner : get_title_4_pos);

        var htmls = ['<tr><td>'];
        for (var i = 0, len = resp.sents.length ; i < len; i++) {
            var o = resp.sents[i];
            if (o.error && o.error.errorMessage) {
                htmls.push( $('<div>').addClass('error bold').text(o.error.errorMessage + '          (' + o.error.fullErrorMessage + ')')[ 0 ].outerHTML );
            } else {
                htmls.push('<div class="' + ((i % 2) === 0 ? 'even' : 'odd') + '">');
                for (var j = 0, ts = o.tuples, len_2 = ts.length; j < len_2; j++) {
                    var t = ts[ j ],
                        $span = $('<span>').attr('label', t.label).addClass(t.label).text(t.word).attr('title', get_title(t.label));
                    htmls.push($span[ 0 ].outerHTML);
                    htmls.push(' ');
                }
                htmls.push('</div><br/>');
            }
        }
        htmls.push('<br/><br/></td></tr>');
        $result.html( htmls.join('') );
    };

    function processing_start() {
        $('#text').addClass('no-change').attr('readonly', 'readonly').attr('disabled', 'disabled');
        $('.result-info').show().removeClass('error').removeClass('cls-undefined').html('processing... <label id="processingTickLabel"></label>');
        $('#processButton, #resetText2Default, #modelType, label[for="modelType"]').hide();//.addClass('disabled');
        $('#processResult tbody').empty();

        processingTickCount = 1;
        setTimeout(processing_tick, 1000);
    };
    function processing_end() {
        $('#text').removeClass('no-change').removeAttr('readonly').removeAttr('disabled');        
        $('#processButton, #resetText2Default, #modelType, label[for="modelType"]').show();
        $('.result-info').removeClass('error').text('');
    };
    function processing_end_without_error() { processing_end(); $('.result-info').hide(); };
    function processing_end_with_error(msg) { processing_end(); $('.result-info').addClass('error').append( $('<span>').text( msg ) ); };
    function processing_end_with_undefined() { processing_end(); $('.result-info').addClass('cls-undefined').append( $('<span>').text('PosTagger for text is undefined.') ); };
    function trim_text(text) { return (text.replace(/(^\s+)|(\s+$)/g, '')); };
    function is_text_empty(text) { return (text.replace(/(^\s+)|(\s+$)/g, '') === ''); };

    var processingTickCount = 1;
    function processingTickCount_to_text( ptc ) {
        var n2 = function (n) { n = n.toString(); return ((n.length === 1) ? ('0' + n) : n); }
        var d  = new Date(new Date(new Date(new Date().setHours(0)).setMinutes(0)).setSeconds( ptc ));
        var t  = n2(d.getHours()) + ':' + n2(d.getMinutes()) + ':' + n2(d.getSeconds()); //d.toLocaleTimeString();
        return (t);
    };
    function processing_tick() {
        var t = processingTickCount_to_text( processingTickCount );
        var $s = $('#processingTickLabel');
        if ($s.length) {
            $s.text(t);
            processingTickCount++;
            setTimeout(processing_tick, 1000);
        } else {
            processingTickCount = 1;
        }
    };
});

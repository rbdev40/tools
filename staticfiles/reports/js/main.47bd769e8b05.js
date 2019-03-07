

(function () {

    'use strict';

    $(document).ready(function () {
    
        window.showSpinner = function () {
            
            if (!$('form.dirty')[0]) {
                $('#spinner-modal').modal('show');
            
                // window.setTimeout(function () {
                //     $('#spinner-modal').modal('hide');
                // }, 10000);
            
                $(window).unload(function() {
                    $('#spinner-modal').modal('hide');
                });
            }
            
        };

    });

}());

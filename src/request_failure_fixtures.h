#ifndef REQUEST_FAILURE_FIXTURES_H
#define REQUEST_FAILURE_FIXTURES_H

/* Minimized requests retained from run_0284, games 30 and 130. */
static const char* const lone_cresselia_request =
    "{\"active\":["
    "{\"moves\":[{\"id\":\"superpower\",\"pp\":6,\"target\":\"normal\",\"disabled\":false}]},"
    "{\"moves\":[{\"id\":\"lunarblessing\",\"pp\":0,\"target\":\"allies\",\"disabled\":true},"
    "{\"id\":\"helpinghand\",\"pp\":28,\"target\":\"adjacentAlly\",\"disabled\":false},"
    "{\"id\":\"psychic\",\"pp\":0,\"target\":\"normal\",\"disabled\":true},"
    "{\"id\":\"icywind\",\"pp\":0,\"target\":\"allAdjacentFoes\",\"disabled\":true}],"
    "\"canTerastallize\":\"Electric\"}],\"side\":{\"id\":\"p1\",\"pokemon\":["
    "{\"ident\":\"p1: Deoxys\",\"details\":\"Deoxys-Speed, L85\",\"condition\":\"0 fnt\",\"active\":true},"
    "{\"ident\":\"p1: Cresselia\",\"details\":\"Cresselia, L80, F\",\"condition\":\"8/323\",\"active\":true},"
    "{\"ident\":\"p1: Urshifu\",\"details\":\"Urshifu, L75, M\",\"condition\":\"0 fnt\",\"active\":false},"
    "{\"ident\":\"p1: Eternatus\",\"details\":\"Eternatus, L70\",\"condition\":\"0 fnt\",\"active\":false},"
    "{\"ident\":\"p1: Hitmonlee\",\"details\":\"Hitmonlee, L86, M\",\"condition\":\"0 fnt\",\"active\":false},"
    "{\"ident\":\"p1: Tinkaton\",\"details\":\"Tinkaton, L82, F\",\"condition\":\"0 fnt\",\"active\":false}]}}";

static const char* const rabsca_revives_active_request =
    "{\"forceSwitch\":[false,true],\"side\":{\"id\":\"p2\",\"pokemon\":["
    "{\"ident\":\"p2: Klawf\",\"details\":\"Klawf, L91, M\",\"condition\":\"0 fnt\",\"active\":true},"
    "{\"ident\":\"p2: Rabsca\",\"details\":\"Rabsca, L87, M\",\"condition\":\"184/272\",\"active\":true,\"reviving\":true},"
    "{\"ident\":\"p2: Roaring Moon\",\"details\":\"Roaring Moon, L76\",\"condition\":\"285/285\",\"active\":false},"
    "{\"ident\":\"p2: Mew\",\"details\":\"Mew, L84\",\"condition\":\"305/305\",\"active\":false},"
    "{\"ident\":\"p2: Toucannon\",\"details\":\"Toucannon, L87, M\",\"condition\":\"281/281\",\"active\":false},"
    "{\"ident\":\"p2: Dodrio\",\"details\":\"Dodrio, L85, M\",\"condition\":\"241/241\",\"active\":false}]},\"noCancel\":true}";

#endif
